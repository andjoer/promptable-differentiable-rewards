#!/usr/bin/env python3

"""
Benchmark anti-periodic losses on two folders: artifacts/ and no_artifacts/.

Outputs:
  1) Per-image CSV with losses per configuration
  2) Summary CSV with means per group (artifacts vs clean) and artifact/clean ratio

Example:
  python benchmark_anti_periodic.py \
    --artifacts_dir artifacts \
    --clean_dir no_artifacts \
    --resize 512 \
    --period 16 --max_harm 6 --sigma_bins 2.0 \
    --window 128 --hop 32 --edge_penalty 8.0 \
    --template_16 T16.png \
    --out per_image_losses.csv --summary_out summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch

from anti_periodic_loss import (
    build_comb_mask_from_period,
    comb_fft_loss,
    comb_fft_ratio_loss,
    comb_local_loss,
    comb_local_ratio_loss,
    comb_local_adaptive_loss,
    template_matched_loss,
    comb_guard_ratio_dc_safe,
    selective_comb_local_loss,
    grid_cosine_projection_loss,
    comb_amplitude_hinge_local,
)


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path) -> List[Path]:
    return [p for p in sorted(d.rglob("*")) if p.suffix.lower() in ALLOWED_EXTS]


def load_img(path: Path, resize: int | None, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize is not None and resize > 0:
        img = img.resize((resize, resize), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return x.to(device)


def maybe_load_template(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] template not found: {p}")
        return None
    im = Image.open(p).convert("L").resize((16, 16), Image.BICUBIC)
    t = np.asarray(im).astype("float32")
    return t


def compute_losses(
    x: torch.Tensor,
    *,
    period: int,
    max_harm: int,
    sigma_bins: float,
    window: int,
    hop: int,
    edge_penalty: float,
    template16: np.ndarray | None,
    tau: float | None,
    combo_mode: str,
    w_local: float,
    tau_global: float | None,
    tau_local: float | None,
    selective_top_p: float,
    selective_coverage_alpha: float,
    proj_harmonics: tuple[int, ...],
    proj_top_p: float,
    proj_detach_norm: bool,
) -> dict:
    # x: (1,3,H,W) in [0,1]
    _, _, H, W = x.shape
    device = x.device
    with torch.no_grad():
        # Global comb (with prebuilt mask for reproducibility)
        M = build_comb_mask_from_period(H=H, W=W, period=period, max_harm=max_harm,
                                        sigma_bins=sigma_bins, vh_weight=(1.0, 1.0), device=device)
        loss_fft_global = comb_fft_loss(x, period=period, mask=M, max_harm=max_harm,
                                        sigma_bins=sigma_bins, vh_weight=(1.0, 1.0), tau=tau)

        # Local comb
        loss_local = comb_local_loss(x, period=period, window=window, hop=hop,
                                     max_harm=max_harm, sigma_bins=sigma_bins, tau=tau)

        # Local adaptive comb
        loss_local_adapt = comb_local_adaptive_loss(
            x, period=period, window=window, hop=hop,
            max_harm=max_harm, sigma_bins=sigma_bins, edge_penalty=edge_penalty, tau=tau
        )

        # Template matched (optional)
        if template16 is not None:
            loss_template = template_matched_loss(x, template16, period=period, tau=tau)
            val_template = float(loss_template.item())
        else:
            val_template = float("nan")

        # Combined metric
        combo_val: float | None
        if combo_mode == "ratio":
            gl = comb_fft_ratio_loss(x, period=period, max_harm=max_harm,
                                     sigma_bins=sigma_bins, use_hann=True, tau=tau_global)
            loc = comb_local_ratio_loss(x, period=period, window=window, hop=hop,
                                        max_harm=max_harm, sigma_bins=sigma_bins, tau=tau_local)
            combo_val = float((gl + w_local * loc).item())
        elif combo_mode == "abs":
            gl = comb_fft_loss(x, period=period, mask=M, max_harm=max_harm,
                               sigma_bins=sigma_bins, vh_weight=(1.0, 1.0), tau=tau_global)
            loca = comb_local_adaptive_loss(
                x, period=period, window=window, hop=hop,
                max_harm=max_harm, sigma_bins=sigma_bins, edge_penalty=edge_penalty, tau=tau_local
            )
            combo_val = float((gl + w_local * loca).item())
        else:
            combo_val = float("nan")

        loss_guard_dc_safe = comb_guard_ratio_dc_safe(
            x,
            period=period,
            max_harm=max_harm,
            sigma_bins=sigma_bins,
            guard_mul=3.0,
            lowfreq_cut=0.02,
            use_hann=True,
            reduction="mean",
            tau=None,
        )

        loss_selective = selective_comb_local_loss(
            x,
            period=period,
            max_harm=max_harm,
            window=window,
            hop=hop,
            sigma_bins=sigma_bins,
            ratio_margin=0.0,
            rel_margin=0.002,
            edge_penalty=edge_penalty,
            top_p=selective_top_p,
            use_hann=True,
            coverage_alpha=selective_coverage_alpha,
            return_parts=False,
        )

        loss_proj = grid_cosine_projection_loss(
            x,
            period=period,
            harmonics=proj_harmonics,
            window=window,
            hop=hop,
            use_hann=True,
            edge_penalty=edge_penalty,
            top_p=proj_top_p,
            detach_norm=proj_detach_norm,
            return_parts=False,
        )

        loss_amp_hinge = comb_amplitude_hinge_local(
            x,
            period=period,
            max_harm=max_harm,
            window=window,
            hop=hop,
            sigma_bins=sigma_bins,
            amp_margin=1e-3,
            use_hann=True,
        )

        return dict(
            fft_global=float(loss_fft_global.item()),
            local=float(loss_local.item()),
            local_adaptive=float(loss_local_adapt.item()),
            template=float(val_template),
            combo=float(combo_val),
            guard_dc_safe=float(loss_guard_dc_safe.item()),
            selective_local=float(loss_selective.item()),
            proj_cosine=float(loss_proj.item()),
            amp_hinge=float(loss_amp_hinge.item()),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifact", type=str)
    ap.add_argument("--clean_dir", default="no_artifact", type=str)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--period", type=int, default=16)
    ap.add_argument("--max_harm", type=int, default=6)
    ap.add_argument("--sigma_bins", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--hop", type=int, default=32)
    ap.add_argument("--edge_penalty", type=float, default=8.0)
    ap.add_argument("--template_16", type=str, default=None, help="optional 16x16 template path")
    ap.add_argument("--tau", type=float, default=None, help="hinge threshold: zero gradients below this value")
    ap.add_argument("--combo_mode", type=str, default="ratio", choices=["ratio", "abs"],
                    help="combine global+local using ratio or absolute energy")
    ap.add_argument("--w_local", type=float, default=0.2, help="weight for local term in combo")
    ap.add_argument("--tau_global", type=float, default=None, help="tau for global term in combo")
    ap.add_argument("--tau_local", type=float, default=None, help="tau for local term in combo")
    # new loss params
    ap.add_argument("--selective_top_p", type=float, default=0.35, help="top-p windows for selective local loss")
    ap.add_argument("--selective_coverage_alpha", type=float, default=0.5, help="blend between top-p and any-above-threshold")
    ap.add_argument("--proj_harmonics", type=str, default="1,2,3", help="comma-separated harmonics, e.g., '1,2,3'")
    ap.add_argument("--proj_top_p", type=float, default=0.35, help="top-p windows for cosine projection loss")
    ap.add_argument("--no_proj_detach_norm", action="store_true", help="disable detached normalization in projection loss")
    ap.add_argument("--out", type=str, default="anti_periodic_per_image.csv")
    ap.add_argument("--summary_out", type=str, default="anti_periodic_summary.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    art_dir = Path(args.artifacts_dir)
    cln_dir = Path(args.clean_dir)
    assert art_dir.is_dir(), f"artifacts_dir not found: {art_dir}"
    assert cln_dir.is_dir(), f"clean_dir not found: {cln_dir}"

    template16 = maybe_load_template(args.template_16)

    # parse harmonics
    try:
        proj_harm = tuple(int(s) for s in args.proj_harmonics.split(",") if s.strip())
        if len(proj_harm) == 0:
            proj_harm = (1, 2, 3)
    except Exception:
        proj_harm = (1, 2, 3)

    rows: List[dict] = []
    for group, root in [("artifacts", art_dir), ("clean", cln_dir)]:
        for p in list_images(root):
            try:
                x = load_img(p, resize=args.resize, device=device)
                losses = compute_losses(
                    x,
                    period=args.period,
                    max_harm=args.max_harm,
                    sigma_bins=args.sigma_bins,
                    window=args.window,
                    hop=args.hop,
                    edge_penalty=args.edge_penalty,
                    template16=template16,
                    tau=args.tau,
                    combo_mode=args.combo_mode,
                    w_local=args.w_local,
                    tau_global=args.tau_global,
                    tau_local=args.tau_local,
                    selective_top_p=args.selective_top_p,
                    selective_coverage_alpha=args.selective_coverage_alpha,
                    proj_harmonics=proj_harm,
                    proj_top_p=args.proj_top_p,
                    proj_detach_norm=(not args.no_proj_detach_norm),
                )
                rows.append({
                    "group": group,
                    "image": p.name,
                    **losses,
                })
            except Exception as e:
                print(f"[warn] failed on {p}: {e}")
                continue

    if not rows:
        print("No valid images processed. Check folders and image formats.")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "image"]).reset_index(drop=True)
    print("\nPer-image losses:")
    print(df.to_string(index=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved per-image: {args.out}")

    # Summary: mean per group and ratio
    def _agg(col: str) -> Tuple[float, float, float]:
        sub = df[["group", col]].dropna()
        m_art = float(sub.loc[sub.group=="artifacts", col].mean()) if (sub.group=="artifacts").any() else float("nan")
        m_cln = float(sub.loc[sub.group=="clean", col].mean()) if (sub.group=="clean").any() else float("nan")
        ratio = (m_art / m_cln) if (not np.isnan(m_art) and not np.isnan(m_cln) and m_cln!=0) else float("nan")
        return m_art, m_cln, ratio

    cols = [
        "fft_global",
        "local",
        "local_adaptive",
        "template",
        "combo",
        "guard_dc_safe",
        "selective_local",
        "proj_cosine",
        "amp_hinge",
    ]
    summ_rows = []
    for c in cols:
        m_art, m_cln, ratio = _agg(c)
        summ_rows.append({
            "loss": c,
            "mean_artifacts": m_art,
            "mean_clean": m_cln,
            "artifact/clean": ratio,
        })
    df_sum = pd.DataFrame(summ_rows)
    print("\nSummary (means and ratio):")
    print(df_sum.to_string(index=False))
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(args.summary_out, index=False)
    print(f"\nSaved summary: {args.summary_out}")


if __name__ == "__main__":
    main()


