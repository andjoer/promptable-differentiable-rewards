#!/usr/bin/env python3
"""
Differentiable photorealism scorer using AIDE model.

Wraps the AIDE detector (https://github.com/shilinyan99/AIDE) as a differentiable
reward function for real vs AI-generated image classification.

Features:
- Differentiable preprocessing pipeline
- Gradients flow through to input images  
- PMI calibration against strict preprocessing
- Support for both tensor and file inputs

Usage:
    scorer = make_aide_reward_fn(
        aide_repo="/path/to/AIDE",
        resume="/path/to/checkpoint.pth",
        device="cuda"
    )
    
    realness, aux = scorer(images, prompts, metadata)
"""
from __future__ import annotations

from typing import Tuple, Sequence
from pathlib import Path
import argparse
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def _to_device_dtype(x: torch.Tensor, device, dtype):
    return x.to(device=device, dtype=dtype)


def _build_pack_from_images(images: torch.Tensor, token_size: int,
                            dinov2_mean: torch.Tensor, dinov2_std: torch.Tensor) -> torch.Tensor:
    """Build AIDE input pack with differentiable preprocessing.
    
    Returns (B,5,3,H,W) tensor with 4 high-pass variants + 1 normalized token.
    """
    B, C, H, W = images.shape
    x01 = ((images.clamp(-1, 1) + 1.0) / 2.0)
    x01r = F.interpolate(x01, size=(token_size, token_size), mode="bilinear", align_corners=False)

    # token normalized (dinov2)
    tokens = (x01r - dinov2_mean) / dinov2_std

    # High-pass filter using Laplacian kernel
    lap_kernel = torch.tensor([[0.0, -1.0, 0.0],
                               [-1.0, 4.0, -1.0],
                               [0.0, -1.0, 0.0]], device=x01r.device, dtype=x01r.dtype)
    lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    
    hp = F.conv2d(x01r, lap_kernel, bias=None, stride=1, padding=1, groups=C)
    hp_abs = hp.abs()

    # Normalize per-sample
    eps = 1e-6
    hp_min = hp_abs.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)
    hp_max = hp_abs.view(B, C, -1).max(dim=-1)[0].view(B, C, 1, 1)
    hp_norm = (hp_abs - hp_min) / (hp_max - hp_min + eps)

    # Create 4 high-pass variants with different intensities
    x_minmin = (hp_norm * 0.05).clamp(0.0, 1.0)
    x_maxmax = (hp_norm * 0.90 - 0.45 + x01r * 0.10).clamp(0.0, 1.0)
    x_minmin1 = (hp_norm * 0.02).clamp(0.0, 1.0)
    x_maxmax1 = (hp_norm * 0.60 - 0.30 + x01r * 0.10).clamp(0.0, 1.0)

    # Apply same normalization as tokens to all HPF branches
    x_minmin = (x_minmin - dinov2_mean) / dinov2_std
    x_maxmax = (x_maxmax - dinov2_mean) / dinov2_std
    x_minmin1 = (x_minmin1 - dinov2_mean) / dinov2_std
    x_maxmax1 = (x_maxmax1 - dinov2_mean) / dinov2_std

    pack = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, tokens], dim=1)  # (B,5,3,Ht,Wt)
    return pack


def _tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach()
    return {
        "shape": tuple(x.shape),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
    }


def _save_image(t: torch.Tensor, path: Path) -> None:
    # t: (3,H,W) float
    td = t.detach().cpu()
    tmin = float(td.min().item())
    tmax = float(td.max().item())
    if tmax - tmin < 1e-6:
        td = torch.zeros_like(td)
    else:
        td = (td - tmin) / (tmax - tmin)
    arr = (td.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def make_aide_reward_fn(
    aide_repo: str,
    resume: str,
    *,
    resnet_path: str | None = None,
    convnext_path: str | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    token_size: int = 256,
    calibrate_to_strict: bool = False,
):
    device = device if isinstance(device, torch.device) else torch.device(device)
    if device.type in ("cpu", "mps"):
        dtype = torch.float32

    aide_path = Path(aide_repo).expanduser().resolve()
    if not aide_path.exists():
        raise FileNotFoundError(f"AIDE repo not found: {aide_path}")
    sys.path.insert(0, str(aide_path))
    import models.AIDE as AIDE  

    model = AIDE.__dict__["AIDE"](resnet_path=resnet_path, convnext_path=convnext_path)
    model.eval().to(device)
    model.requires_grad_(False)
    param_dtype = next(model.parameters()).dtype

    ckpt = torch.load(str(resume), map_location="cpu")
    candidates = [ckpt.get("model"), ckpt.get("model_ema"), ckpt.get("state_dict"), ckpt]
    for state in candidates:
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            break

    dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    dinov2_std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)

    # Optional strict pipeline (no-grad) for calibration toward main_finetune outputs
    strict_pack_builder_from_tensor = None
    strict_pack_builder_from_path = None
    if calibrate_to_strict:
        try:
            from data.datasets import transform_before_test, transform_train  
            from data.dct import DCT_base_Rec_Module  
        except Exception as e:
            raise RuntimeError("Failed to import AIDE strict preprocessing modules for calibration.") from e
        dct_module = DCT_base_Rec_Module()

        def _build_pack_strict_from_tensor(x: torch.Tensor) -> torch.Tensor:
            # x: (3,H,W) in [-1,1] on device
            x01 = ((x.detach().cpu().clamp(-1,1) + 1.0) / 2.0)
            arr = (x01.permute(1,2,0).numpy() * 255.0).astype('uint8')
            img = Image.fromarray(arr)
            img_t = transform_before_test(img)
            xm0, xM0, xm1, xM1 = dct_module(img_t)
            x_0 = transform_train(img_t)
            xm0 = transform_train(xm0)
            xM0 = transform_train(xM0)
            xm1 = transform_train(xm1)
            xM1 = transform_train(xM1)
            pack = torch.stack([xm0, xM0, xm1, xM1, x_0], dim=0).unsqueeze(0).to(device)
            return pack
        strict_pack_builder_from_tensor = _build_pack_strict_from_tensor

        def _build_pack_strict_from_path(path: Path) -> torch.Tensor:
            img = Image.open(path).convert('RGB')
            img_t = transform_before_test(img)
            xm0, xM0, xm1, xM1 = dct_module(img_t)
            x_0 = transform_train(img_t)
            xm0 = transform_train(xm0)
            xM0 = transform_train(xM0)
            xm1 = transform_train(xm1)
            xM1 = transform_train(xM1)
            pack = torch.stack([xm0, xM0, xm1, xM1, x_0], dim=0).unsqueeze(0).to(device)
            return pack
        strict_pack_builder_from_path = _build_pack_strict_from_path

    @torch.enable_grad()
    def scorer(images: torch.Tensor, _prompts: Tuple[str, ...], _meta=None):
        images = images.to(device=device, dtype=dtype)
        pack = _build_pack_from_images(images, token_size, dinov2_mean, dinov2_std)

        # Pre-logits per-branch affine calibration in input space (match strict stats) to reduce instability
        if calibrate_to_strict and (strict_pack_builder_from_tensor is not None or strict_pack_builder_from_path is not None):
            with torch.no_grad():
                B = images.shape[0]
                use_paths = False
                paths: Sequence[Path] | None = None
                if _meta is not None and isinstance(_meta, dict) and "paths" in _meta:
                    try:
                        paths = tuple(Path(p) for p in _meta["paths"]) 
                        if len(paths) == B:
                            use_paths = True
                    except Exception:
                        use_paths = False

                strict_packs = []
                for i in range(B):
                    if use_paths and strict_pack_builder_from_path is not None and paths is not None:
                        sp = strict_pack_builder_from_path(paths[i])  # (1,5,3,H,W)
                    else:
                        sp = strict_pack_builder_from_tensor(images[i])  
                    strict_packs.append(sp)
                strict_pack = torch.cat(strict_packs, dim=0).to(dtype=pack.dtype)  # (B,5,3,H,W)

                # Compute per-sample, per-branch, per-channel mean/std over spatial dims
                def _ms(t: torch.Tensor):
                    mean = t.mean(dim=(-1, -2))  # (B,5,3)
                    std = t.std(dim=(-1, -2)).clamp_min(1e-6)  # (B,5,3)
                    return mean, std

                mean_diff, std_diff = _ms(pack)
                mean_strict, std_strict = _ms(strict_pack)

                # Only calibrate first 4 branches; leave tokens unchanged
                scale = std_strict[:, :4, :] / std_diff[:, :4, :]
                shift = mean_strict[:, :4, :] - scale * mean_diff[:, :4, :]
                scale = scale.view(B, 4, 3, 1, 1)
                shift = shift.view(B, 4, 3, 1, 1)
                pack[:, :4] = (pack[:, :4] * scale) + shift

        pack = pack.to(dtype=param_dtype)
        logits = model(pack)

        # Detect invalid differentiable predictions per-sample (NaN in logits)
        invalid_diff_mask = torch.isnan(logits).any(dim=1)

        probs = logits.float().softmax(dim=-1)
        # AIDE uses class 0 = real, class 1 = fake
        realness_diff = probs[:, 0]
        realness_diff = torch.nan_to_num(realness_diff, nan=0.5, posinf=1.0, neginf=0.0)

        if strict_pack_builder_from_tensor is None and strict_pack_builder_from_path is None:
            return realness_diff, {}

        # Shadow strict evaluation (no grad) for calibration
        with torch.no_grad():
            strict_scores = []
            strict_invalid_mask_list = [] 
            use_paths = False
            paths: Sequence[Path] | None = None
            if _meta is not None and isinstance(_meta, dict) and "paths" in _meta:
                try:
                    paths = tuple(Path(p) for p in _meta["paths"])
                    if len(paths) == images.shape[0]:
                        use_paths = True
                except Exception:
                    use_paths = False

            for i in range(images.shape[0]):
                if use_paths and paths is not None:
                    spack = strict_pack_builder_from_path(paths[i])
                else:
                    spack = strict_pack_builder_from_tensor(images[i]) 
                slogits = model(spack.to(dtype=param_dtype))
                if torch.isnan(slogits).any():
                    # If strict also produces NaN, mark invalid and fall back to 0.5
                    strict_invalid_mask_list.append(True)
                    strict_scores.append(0.5)
                else:
                    sprob = torch.softmax(slogits, dim=-1)[0, 0].item()
                    strict_invalid_mask_list.append(False)
                    strict_scores.append(sprob)
            strict = torch.tensor(strict_scores, device=realness_diff.device, dtype=realness_diff.dtype)
            strict_invalid_mask = torch.tensor(strict_invalid_mask_list, device=realness_diff.device, dtype=torch.bool)

        # Fit an affine map y ≈ a*x + b using only valid pairs, apply per-sample fallback
        x = realness_diff  # (B,)
        y = strict         # (B,)
        valid_pair_mask = (~invalid_diff_mask) & (~strict_invalid_mask)
        num_valid_pairs = int(valid_pair_mask.sum().item())

        # Default to using strict scores if we cannot fit a reliable calibration
        use_strict_only = num_valid_pairs < 2

        if not use_strict_only:
            xv = x[valid_pair_mask]
            yv = y[valid_pair_mask]
            xv_mean = xv.mean()
            yv_mean = yv.mean()
            cov = ((xv - xv_mean) * (yv - yv_mean)).mean()
            var = ((xv - xv_mean) ** 2).mean() + 1e-8
            a = cov / var
            b = yv_mean - a * xv_mean
            calibrated_all = (a * x + b).clamp(0.0, 1.0)
        else:
            calibrated_all = y

        # Per-sample robust merge:
        # - If differentiable path invalid for sample => use strict (if strict invalid too, 0.5 already)
        # - Else if strict-only mode => use strict
        # - Else use calibrated value
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            if invalid_diff_mask[i]:
                out[i] = y[i]
            elif use_strict_only:
                out[i] = y[i]
            else:
                out[i] = calibrated_all[i]

        # Warnings for non-differentiable fallbacks
        non_diff_mask = invalid_diff_mask.clone()
        if use_strict_only:
            non_diff_mask[:] = True

        if bool(non_diff_mask.any().item()):
            names = None
            if _meta is not None and isinstance(_meta, dict) and "paths" in _meta:
                try:
                    names = [str(p) for p in _meta["paths"]]
                except Exception:
                    names = None
            idxs = [int(i) for i in torch.nonzero(non_diff_mask, as_tuple=False).view(-1).tolist()]
            if names and len(names) == len(non_diff_mask):
                affected = ", ".join(names[i] for i in idxs)
            else:
                affected = ", ".join(str(i) for i in idxs)
            if use_strict_only:
                print(f"WARNING: AIDE scorer used strict outputs (non-differentiable) for entire batch. Affected: {affected}")
            else:
                print(f"WARNING: AIDE scorer fell back to strict outputs (non-differentiable) for samples: {affected}")

        return out, {"realness_diff": realness_diff.detach(), "realness_strict": strict}

    return scorer


def _load_img(path: Path, device: torch.device, dtype: torch.dtype, H=256, W=256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((W, H))
    arr = torch.from_numpy(np.asarray(img).astype("float32") / 255.0).permute(2, 0, 1)
    t = (arr * 2 - 1).to(device=device, dtype=dtype).requires_grad_(True)
    return t


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--aide_repo", default="AIDE")
    ap.add_argument("--resume", default="AIDE/GenImage_train.pth")
    ap.add_argument("--resnet_path", default="None")
    ap.add_argument("--convnext_path", default="None")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--token_size", type=int, default=256)
    ap.add_argument("--strict_pack", action="store_true", help="Replicate main_finetune preprocessing and print image_path,realness,fakeness,pred_label")
    ap.add_argument("--calibrate_to_strict", action="store_true", help="Calibrate differentiable scores to match strict preprocessing output using affine transformation")
    ap.add_argument("--compare_preproc", action="store_true", help="Compare diff vs strict preprocessing tensors and dump images")
    ap.add_argument("--dump_dir", default="preproc_debug", help="Directory to dump comparison images")
    # Gradient visualization 
    ap.add_argument("--viz_grads", action="store_true", help="Compute and save gradient visualizations per image")
    ap.add_argument("--viz_outdir", default="debug_images", help="Directory to save gradient visualizations")
    ap.add_argument("--viz_alpha", type=float, default=0.6, help="Overlay strength (0..1)")
    ap.add_argument("--viz_norm", choices=["percentile", "log", "none"], default="percentile", help="Dynamic range compression")
    ap.add_argument("--viz_pmin", type=float, default=2.0, help="Lower percentile for percentile norm (0..100)")
    ap.add_argument("--viz_pmax", type=float, default=98.0, help="Upper percentile for percentile norm (0..100)")
    ap.add_argument("--viz_log_alpha", type=float, default=10.0, help="Alpha for log compression (higher=more compression)")
    ap.add_argument("--viz_blur", type=float, default=0.0, help="Gaussian blur sigma (px) for smoothing the heatmap")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available(): # no good results on mps
            device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if device.type in ("cpu", "mps"):
        dtype = torch.float32

    if args.strict_pack:
        aide_path = Path(args.aide_repo).expanduser().resolve()
        if not aide_path.exists():
            raise FileNotFoundError(f"AIDE repo not found: {aide_path}")
        sys.path.insert(0, str(aide_path))
        import models.AIDE as AIDE 
        from data.datasets import transform_before_test, transform_train 
        from data.dct import DCT_base_Rec_Module 
        resnet_path = args.resnet_path if args.resnet_path != "None" and Path(str(args.resnet_path)).exists() else None
        convnext_path = args.convnext_path if args.convnext_path != "None" and Path(str(args.convnext_path)).exists() else None
        model = AIDE.__dict__["AIDE"](resnet_path=resnet_path, convnext_path=convnext_path)
        model.eval().to(device)
        ckpt = torch.load(str(args.resume), map_location="cpu")
        state = ckpt.get("model") or ckpt.get("model_ema") or ckpt.get("state_dict") or ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

        dct_module = DCT_base_Rec_Module()

        for img_path in args.images:
            p = Path(img_path)
            img = Image.open(p).convert('RGB')
            img_t = transform_before_test(img)
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_t)
            x_0 = transform_train(img_t)
            x_minmin = transform_train(x_minmin)
            x_maxmax = transform_train(x_maxmax)
            x_minmin1 = transform_train(x_minmin1)
            x_maxmax1 = transform_train(x_maxmax1)
            pack = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(pack)
                probs = torch.softmax(logits, dim=1)[0].detach().cpu()
                realness = float(probs[0].item())
                fakeness = float(probs[1].item())
                pred_label = int(fakeness > 0.5)
                print(f"{p.name},{realness},{fakeness},{pred_label}")
        return

    scorer = make_aide_reward_fn(
        aide_repo=args.aide_repo,
        resume=args.resume,
        resnet_path=args.resnet_path if Path(str(args.resnet_path)).exists() else None,
        convnext_path=args.convnext_path if Path(str(args.convnext_path)).exists() else None,
        device=device,
        dtype=dtype,
        token_size=args.token_size,
        calibrate_to_strict=args.calibrate_to_strict,
    )

    ims = [Path(p) for p in args.images]
    batch = torch.stack([_load_img(p, device, dtype) for p in ims], 0)
    prompts = tuple(["" for _ in ims])
    scores, _ = scorer(batch, prompts, {"paths": [str(p) for p in ims]})
    for p, s in zip(ims, scores):
        real = float(s.item())
        fake = 1.0 - real
        pred = int(fake > 0.5)
        print(f"{p.name},{real},{fake},{pred}")

    # Optional gradient visualization (save input, heatmap, overlay)
    if args.viz_grads:
        from datetime import datetime
        out_root = Path(args.viz_outdir) / ("aide_gradviz_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        out_root.mkdir(parents=True, exist_ok=True)

        def _gaussian_blur_hw(x: torch.Tensor, sigma: float) -> torch.Tensor:
            if sigma <= 0:
                return x
            radius = int(max(1, round(3.0 * sigma)))
            ksize = 2 * radius + 1
            grid = torch.arange(ksize, dtype=torch.float32) - radius
            kernel_1d = torch.exp(-0.5 * (grid / sigma) ** 2)
            kernel_1d = (kernel_1d / kernel_1d.sum()).view(1, 1, -1)
            x4 = x.view(1, 1, x.shape[0], x.shape[1])
            pad = (radius, radius, 0, 0)
            xh = F.conv2d(F.pad(x4, pad, mode="reflect"), kernel_1d.unsqueeze(2))
            pad = (0, 0, radius, radius)
            xv = F.conv2d(F.pad(xh, pad, mode="reflect"), kernel_1d.unsqueeze(3))
            return xv.view_as(x)

        def _compress(g_map: torch.Tensor) -> torch.Tensor:
            if args.viz_blur > 0.0:
                g_map = _gaussian_blur_hw(g_map, float(args.viz_blur))
            if args.viz_norm == "percentile":
                flat = g_map.flatten()
                lo = torch.quantile(flat, float(args.viz_pmin) / 100.0)
                hi = torch.quantile(flat, float(args.viz_pmax) / 100.0)
                g = (g_map - lo) / (hi - lo + 1e-12)
                return g.clamp(0, 1)
            elif args.viz_norm == "log":
                m = g_map.max()
                g = g_map / (m + 1e-12)
                g = torch.log1p(float(args.viz_log_alpha) * g) / np.log1p(float(args.viz_log_alpha))
                return g.clamp(0, 1)
            else:
                m = g_map.max()
                return (g_map / (m + 1e-12)).clamp(0, 1)

        # compute grad w.r.t. inputs
        total = scores.sum()
        g = torch.autograd.grad(total, batch, retain_graph=False, allow_unused=True)[0]
        if g is None:
            print("[warn] gradient w.r.t inputs is None; ensure inputs require_grad and model path is differentiable.")
        else:
            imgs = batch.detach().cpu()
            grads = g.detach().cpu()
            for idx, (p, img, gg) in enumerate(zip(ims, imgs, grads)):
                # img: (3,H,W) in [-1,1]
                rgb = ((img.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
                g_map = gg.abs().mean(0)
                g_map = _compress(g_map)
                overlay = rgb.clone()
                overlay[0] = (1 - args.viz_alpha) * overlay[0] + args.viz_alpha * g_map
                overlay[1] = (1 - args.viz_alpha) * overlay[1]
                overlay[2] = (1 - args.viz_alpha) * overlay[2]

                def to_pil(t: torch.Tensor) -> Image.Image:
                    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype("uint8"))

                base = p.stem
                to_pil(rgb).save(out_root / f"{base}_input.png")
                to_pil(g_map.expand_as(rgb)).save(out_root / f"{base}_grad.png")
                to_pil(overlay).save(out_root / f"{base}_overlay.png")


if __name__ == "__main__":
    _cli()


