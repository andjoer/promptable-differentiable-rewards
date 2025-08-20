#!/usr/bin/env python3
"""
AlignProp training with aesthetic and similarity rewards for image2image models (here Flux Kontext).

Supports multiple reward types:
- Aesthetic scoring (InternVL, default aesthetic scorer)
- Image similarity (LPIPS, face embeddings, InternVL content)
- Photorealism detection (AIDE)
- Anti-grid artifact penalties
- Sharpness/brightness anchors
"""

from __future__ import annotations

import datetime as _dt
import itertools
import random
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage

from transformers import HfArgumentParser

from trl import AlignPropConfig, AlignPropTrainerFluxKontext
from trl.models.auxiliary_modules import aesthetic_scorer
from trl.models.modeling_flux_kontext import DDPOFluxKontextPipeline 

from accelerate import PartialState
from accelerate.logging import get_logger

from anti_periodic_loss import (
    comb_fft_ratio_loss,
    comb_guard_ratio_dc_safe,
    selective_comb_local_loss,
    grid_cosine_projection_loss,
    mean_luma_band,
    tenengrad_floor_loss,
)

logger = get_logger(__name__)

# Constants
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AESTHETIC_DEFAULT_ID = "trl-lib/ddpo-aesthetic-predictor"
AESTHETIC_DEFAULT_FILE = "aesthetic-model.pth"


def _save_debug_images(images: torch.Tensor, input_images: torch.Tensor | None = None):
    """Save debug images to timestamped directory."""
    import datetime as _dt
    from torchvision.transforms import ToPILImage
    
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = Path("debug_images") / timestamp
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    to_pil = ToPILImage()
    for i, img in enumerate(images):
        to_pil((img.clamp(-1, 1) + 1) / 2).save(debug_dir / f"generated_{i}.png")
    
    if input_images is not None:
        for i, img in enumerate(input_images):
            to_pil((img.clamp(-1, 1) + 1) / 2).save(debug_dir / f"reference_{i}.png")
    
    logger.info(f"Debug images saved to {debug_dir}")

@dataclass
class ScriptArguments:
    """Configuration for AlignProp training."""
    
    # Model configuration
    pretrained_model: str = field(
        default="black-forest-labs/FLUX.1-Kontext-dev",
        metadata={"help": "Pretrained model repo or local path"},
    )
    pretrained_revision: str = field(default="main")
    use_lora: bool = field(default=True)
    lora_rank: int = field(default=8)

    # Aesthetic reward configuration
    hf_hub_aesthetic_model_id: str = field(default=AESTHETIC_DEFAULT_ID)
    hf_hub_aesthetic_model_filename: str = field(default=AESTHETIC_DEFAULT_FILE)
    aesthetic_model_type: str = field(
        default="aesthetic_scorer",
        metadata={"help": "Aesthetic model type: 'aesthetic_scorer' | 'v2.5' | 'internvl'"},
    )
    aesthetic_reward_scale: float = field(default=0.4)

    # Similarity reward configuration
    similarity_backend: str = field(
        default="lpips", 
        metadata={"help": "Similarity backend: 'lpips' | 'face' | 'internvl-content'"}
    )
    image_reward_scale: float = field(default=0.5)

    # Photorealism reward configuration
    photo_backend: str = field(
        default="none",
        metadata={"help": "Photorealism backend: 'none' | 'aide'"},
    )
    photo_reward_scale: float = field(default=0.0)
    aide_repo: Path = field(default=Path("./AIDE"))
    aide_resume: Path | None = field(default=None)

    # Anti-grid penalty configuration
    grid_ratio_weight: float = field(default=0.0, metadata={"help": "Weight for grid artifact penalty"})
    grid_ratio_period: int = field(default=16)
    grid_ratio_max_harm: int = field(default=3)
    grid_ratio_sigma_bins: float = field(default=0.8)
    grid_ratio_tau: float = field(default=2e-4, metadata={"help": "Hinge threshold for ratio penalty"})
    grid_eval_resize: int = field(default=512, metadata={"help": "Resize images to this size for grid evaluation"})

    # Brightness/contrast anchor configuration
    brightness_anchor_weight: float = field(default=0.2, metadata={"help": "Weight for brightness penalty"})
    brightness_target: float = field(default=0.50, metadata={"help": "Target mean luminance (0.0-1.0)"})
    brightness_band: float = field(default=0.06, metadata={"help": "Tolerance band around target brightness"})
    crispness_anchor_weight: float = field(default=0.2, metadata={"help": "Weight for crispness penalty"})

    # Sharpness reward configuration
    sharpness_backend: str = field(default="none", metadata={"help": "Sharpness backend: 'none' | 'internvl'"})
    sharpness_reward_scale: float = field(default=0.0)

    # Data configuration
    data_dir: Path = field(default=Path("./train_images"))
    prompt_list_file: Path | None = field(default=None)
    style_reference: Path | None = field(
        default=None,
        metadata={"help": "Optional style reference image for InternVL aesthetic scoring"},
    )

    # Debugging
    debug_images: bool = field(default=False, metadata={"help": "Save debug images during training"})


class ImageDataset(Iterable[Path]):
    def __init__(self, root: Path):
        if not root.is_dir():
            raise FileNotFoundError(root)
        self._paths = [p for p in root.rglob("*") if p.suffix.lower() in ALLOWED_EXTS]
        if not self._paths:
            raise RuntimeError(f"No images with {ALLOWED_EXTS} under {root}")
        logger.info("Found %d training images under %s", len(self._paths), root)

    def __iter__(self):
        return itertools.cycle(self._paths)


def load_prompts(path: Path | None) -> List[str]:
    if path is None:
        return ["make it a professional portrait photograph"]
    if not path.is_file():
        raise FileNotFoundError(path)
    prompts = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not prompts:
        raise ValueError("Prompt list file is empty")
    return prompts


def lpips_reward(model: lpips.LPIPS, gen: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return similarity in [0,1] (higher = more similar)."""
    if gen.min() >= 0:
        gen = gen * 2 - 1
        ref = ref * 2 - 1
    with torch.no_grad():
        d = model(gen, ref).squeeze()  # distance
    return torch.exp((-5.0 * d).clamp(max=20)).clamp_min_(1e-4)


def load_facenet(device: torch.device | str) -> torch.nn.Module:
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained="vggface2").eval()
    model.requires_grad_(False)
    return model.to(device)


def facenet_reward(model, gen: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if gen.min() >= 0:
        gen = gen * 2 - 1
        ref = ref * 2 - 1
    gen = F.interpolate(gen, (160, 160), mode="bilinear", align_corners=False)
    ref = F.interpolate(ref, (160, 160), mode="bilinear", align_corners=False)
    gen_emb = F.normalize(model(gen), dim=-1)
    with torch.no_grad():
        ref_emb = F.normalize(model(ref), dim=-1)
    return F.cosine_similarity(gen_emb, ref_emb)


# Reward combination

def _check_gradients(images: torch.Tensor, scores: torch.Tensor, tag: str = "reward"):
    """Debug utility to verify gradient flow."""
    if not scores.requires_grad:
        logger.warning(f"[{tag}] Scores do not require gradients")
        return
    
    try:
        g = torch.autograd.grad(scores.sum(), images, retain_graph=True, allow_unused=True)[0]
        if g is not None:
            nz = g.detach().abs().gt(0).sum().item()
            mean_abs = g.detach().abs().mean().item()
            logger.debug(f"[{tag}] Gradient: shape={g.shape}, nonzeros={nz}, mean_abs={mean_abs:.2e}")
        else:
            logger.warning(f"[{tag}] No gradients computed")
    except Exception as e:
        logger.warning(f"[{tag}] Gradient check failed: {e}")


def combined_reward(
    images: torch.Tensor,
    input_images: torch.Tensor | None,
    prompts: Tuple[str, ...],
    metadata: Tuple[dict, ...],
    *,
    aest_fn,
    sim_fn, 
    scale_aest: float,
    scale_img: float,
    photo_fn=None,
    scale_photo: float = 0.0,
    sharpness_fn=None,
    scale_sharpness: float = 0.0,
    grid_config: dict,
    brightness_config: dict,
    log_debug: bool = False,
):
    """Compute combined reward from multiple components.
    
    Args:
        images: Generated images [B,3,H,W] in [-1,1]
        input_images: Reference images [B,3,H,W] in [-1,1] 
        prompts: Text prompts
        metadata: Per-sample metadata
        aest_fn: Aesthetic scoring function
        sim_fn: Similarity scoring function
        scale_aest: Aesthetic reward weight
        scale_img: Similarity reward weight
        photo_fn: Optional photorealism function
        scale_photo: Photorealism reward weight
        sharpness_fn: Optional sharpness function
        scale_sharpness: Sharpness reward weight
        grid_config: Anti-grid penalty configuration
        brightness_config: Brightness/contrast anchor configuration
        log_debug: Whether to save debug images
        
    Returns:
        Combined reward tensor [B] and metadata dict
    """

    # Compute individual reward components
    aest_score, _ = aest_fn(images, prompts, metadata)
    _check_gradients(images, aest_score, "aesthetic")
    
    images = images.to(dtype=torch.float32)
    
    # Similarity reward
    if input_images is None:
        sim = torch.zeros_like(aest_score)
    else:
        sim = sim_fn(images, input_images.to(dtype=torch.float32, device=images.device))

    # Optional photorealism reward
    if photo_fn is None or scale_photo == 0.0:
        photo = torch.zeros_like(aest_score)
    else:
        photo, _ = photo_fn(images, prompts, metadata)
        _check_gradients(images, photo, "photorealism")

    # Optional sharpness reward
    if sharpness_fn is None or scale_sharpness == 0.0:
        sharp = torch.zeros_like(aest_score)
    else:
        sharp, _ = sharpness_fn(images, prompts, metadata)
        _check_gradients(images, sharp, "sharpness")

    # Anti-grid penalties
    img01 = ((images.clamp(-1, 1) + 1.0) / 2.0).to(dtype=torch.float32)
    
    # Resize for grid evaluation if specified
    grid_eval_size = grid_config.get("eval_resize", 512)
    if grid_eval_size > 0:
        H, W = img01.shape[-2:]
        if (H, W) != (grid_eval_size, grid_eval_size):
            img01 = F.interpolate(
                img01, size=(grid_eval_size, grid_eval_size), 
                mode="bilinear", align_corners=False
            )

    # Grid artifact detection
    period = grid_config.get("period", 16)
    max_harm = grid_config.get("max_harm", 3)
    sigma_bins = grid_config.get("sigma_bins", 0.8)
    tau = grid_config.get("tau", 2e-4)
    
    grid_ratio_raw = comb_fft_ratio_loss(
        img01, period=period, max_harm=max_harm, sigma_bins=sigma_bins,
        use_hann=True, tau=None, reduction="mean"
    )
    
    grid_ratio_hinged = comb_fft_ratio_loss(
        img01, period=period, max_harm=max_harm, sigma_bins=sigma_bins,
        use_hann=True, tau=tau, reduction="mean"
    )

    # Advanced grid penalties
    grid_proj, proj_info = grid_cosine_projection_loss(
        img01, period=period, harmonics=(1, 2, 3),
        window=128, hop=32, use_hann=True, edge_penalty=12.0,
        top_p=0.30, detach_norm=True, return_parts=True
    )

    grid_selective = selective_comb_local_loss(
        img01, period=period, max_harm=max(4, max_harm),
        window=128, hop=32, sigma_bins=1.0, ratio_margin=0.0,
        rel_margin=0.002, edge_penalty=12.0, top_p=0.30,
        use_hann=True, coverage_alpha=0.7
    )

    grid_penalty = grid_proj + 0.5 * grid_selective

    # Brightness and contrast anchors
    bright_target = brightness_config.get("target", 0.50)
    bright_band = brightness_config.get("band", 0.06)
    bright_weight = brightness_config.get("weight", 0.2)
    crisp_weight = brightness_config.get("crisp_weight", 0.2)
    
    bright_penalty = mean_luma_band(img01, target=bright_target, band=bright_band, weight=1.0)
    crisp_penalty = tenengrad_floor_loss(img01, floor=1.5e-3, weight=0.25)
    
    # Combine all reward components
    grid_weight = grid_config.get("weight", 0.0)
    
    reward = (
        scale_aest * aest_score
        + scale_img * sim.to(aest_score.dtype)
        + scale_photo * photo.to(aest_score.dtype)
        + scale_sharpness * sharp.to(aest_score.dtype)
        - grid_weight * grid_penalty.to(aest_score.dtype)
        - bright_weight * bright_penalty.to(aest_score.dtype)
        - crisp_weight * crisp_penalty.to(aest_score.dtype)
    )
    # Collect metrics
    reward_metadata = {
        "aesthetic": aest_score.mean().item(),
        "similarity": sim.mean().item(),
        "photorealism": photo.mean().item(),
        "sharpness": sharp.mean().item(),
        "grid_raw": grid_ratio_raw.item(),
        "grid_penalty": grid_penalty.item(),
        "brightness_penalty": bright_penalty.item(),
        "crispness_penalty": crisp_penalty.item(),
        "grid_eval_size": grid_eval_size,
    }
    
    # Add projection diagnostics
    reward_metadata.update({
        f"proj_{k}": float(v) for k, v in proj_info.items()
    })
    
    # Add guard ratio if available
    try:
        guard_ratio = comb_guard_ratio_dc_safe(
            img01, period=period, max_harm=max_harm, sigma_bins=sigma_bins,
            guard_mul=3.0, lowfreq_cut=0.02, use_hann=True, reduction="mean"
        )
        reward_metadata["grid_guard"] = guard_ratio.item()
    except Exception as e:
        logger.debug(f"Guard ratio computation failed: {e}")


    # Save debug images if requested
    if log_debug:
        _save_debug_images(images, input_images)

    logger.info(
        "Reward components - aesthetic: %.3f, similarity: %.3f, photorealism: %.3f, "
        "sharpness: %.3f, grid_penalty: %.5f, combined: %.3f",
        aest_score.mean().item(), sim.mean().item(), photo.mean().item(),
        sharp.mean().item(), grid_penalty.item(), reward.mean().item()
    )
    
    return reward, reward_metadata


def main():
    parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    sargs, train_cfg = parser.parse_args_into_dataclasses()


    PartialState()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16 if train_cfg.mixed_precision == "fp16" else torch.float32
    if train_cfg.seed is not None:
        torch.manual_seed(train_cfg.seed)
        random.seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)

    # ---------- data -----------------------------------------------
    ds = ImageDataset(sargs.data_dir)
    ds_iter = iter(ds)
    prompts = load_prompts(sargs.prompt_list_file)

    def prompt_fn():
        # robustly fetch a valid reference image; skip corrupted files
        ref_img = None
        img_path = None
        max_tries = max(1, len(ds._paths))
        for _ in range(max_tries):
            img_path = next(ds_iter)
            try:
                ref_img = Image.open(img_path).convert("RGB")
                break
            except Exception as e:  # corrupted or unreadable
                logger.warning("Skipping corrupted image %s: %s", img_path, str(e))
                continue
        if ref_img is None:
            raise RuntimeError("No valid images found; all candidates failed to load.")
        prompt = random.choice(prompts)
        return prompt, None, ref_img, {}

    # ---------- aesthetic model ------------------------------------
    if sargs.aesthetic_model_type.lower() == "v2.5":
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        model, preprocess = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
        model = model.to(device)

        def aest_v25(images, *_):
            img = (images.clamp(-1, 1) + 1) / 2  # to [0,1]
            img = F.interpolate(img, (384, 384), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
            img = (img - mean) / 0.5
            with torch.set_grad_enabled(img.requires_grad):
                return model(img).logits.squeeze(), {}
        aesthetic_fn = aest_v25
 
    elif sargs.aesthetic_model_type.lower() == "internvl":
        # InternVL as aesthetic scorer, with optional shared reuse for sharpness
        from internvl_scorer import (
            make_reward_fn as make_internvl_reward_fn,
            load_img as load_img_internvl,
            load_internvl_core,
        )

        internvl_model_id = "OpenGVLab/InternVL3-14B-hf"
        shared_processor, shared_model = load_internvl_core(
            internvl_model_id, device=device, dtype=dtype
        )

        if sargs.style_reference is not None:
            internvl_style = make_internvl_reward_fn(
                model_id=internvl_model_id,
                task="similarity",
                device=device,
                dtype=dtype,
                temperature=1.0,
                objective="expected",
                calibration="pmi_null",
                margin_temp=1.0,
                entropy_coef=0.0,
                processor=shared_processor,
                model=shared_model,
            )
            ref_path = sargs.style_reference.expanduser()
            if not ref_path.is_file():
                raise FileNotFoundError(f"style_reference not found: {ref_path}")
            ref_img_base = load_img_internvl(ref_path, device=device, dtype=dtype, requires_grad=False)

            def aesthetic_fn(images, *_):
                if images.dim() != 4 or images.size(1) != 3:
                    raise ValueError("Expect images of shape (B,3,H,W)")
                B, _, H, W = images.shape
                ref_resized = F.interpolate(ref_img_base.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
                ref_b = ref_resized.unsqueeze(0).expand(B, -1, -1, -1).to(device=images.device, dtype=images.dtype)
                pair = torch.stack([ref_b, images], dim=1)
                prompts = tuple([""] * B)
                scores, aux = internvl_style(pair, prompts, None)
                return scores, aux
        else:
            internvl_single = make_internvl_reward_fn(
                model_id=internvl_model_id,
                task="single",
                device=device,
                dtype=dtype,
                temperature=1.0,
                objective="expected",
                calibration="pmi_null",
                margin_temp=1.0,
                entropy_coef=0.0,
                processor=shared_processor,
                model=shared_model,
            )

            def aesthetic_fn(images, *_):
                if images.dim() != 4 or images.size(1) != 3:
                    raise ValueError("Expect images of shape (B,3,H,W)")
                B = images.size(0)
                images_batched = images.unsqueeze(1)
                prompts = tuple([""] * B)
                scores, aux = internvl_single(images_batched, prompts, None)
                return scores, aux
    else:
        aesthetic_fn = aesthetic_scorer(sargs.hf_hub_aesthetic_model_id, sargs.hf_hub_aesthetic_model_filename)

    # ---------- similarity model -----------------------------------
    if sargs.similarity_backend == "lpips":
        try:
            import lpips
        except Exception as e:
            raise RuntimeError("lpips not found, please install it with `pip install lpips`") from e

        sim_model = lpips.LPIPS(net="alex").to(device).eval()
        sim_fn = partial(lpips_reward, sim_model)
    elif sargs.similarity_backend == "face":
        sim_model = load_facenet(device)
        sim_fn = partial(facenet_reward, sim_model)
    elif sargs.similarity_backend == "internvl-content":
        # reuse shared InternVL core if already loaded for aesthetic/sharpness
        internvl_model_id = "OpenGVLab/InternVL3-14B-hf"
        try:
            shared_processor  
            shared_model     
        except NameError:
            from internvl_scorer import load_internvl_core as _load_internvl_core
            shared_processor, shared_model = _load_internvl_core(
                internvl_model_id, device=device, dtype=dtype
            )
        from internvl_scorer import make_reward_fn as make_internvl_reward_fn
        content_scorer = make_internvl_reward_fn(
            model_id=internvl_model_id,
            task="content-similarity",
            device=device,
            dtype=dtype,
            temperature=1.0,
            objective="expected",
            calibration="pmi_null",
            margin_temp=1.0,
            entropy_coef=0.0,
            processor=shared_processor,
            model=shared_model,
        )

        def sim_fn(gen: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            if gen.dim() != 4 or gen.size(1) != 3:
                raise ValueError("sim_fn expects (B,3,H,W) tensors")
            if ref is None:
                return torch.zeros(gen.size(0), device=gen.device, dtype=gen.dtype)
            if ref.shape != gen.shape:
                ref = F.interpolate(ref, size=gen.shape[-2:], mode="bilinear", align_corners=False)
            B = gen.size(0)
            pair = torch.stack([gen, ref.to(device=gen.device, dtype=gen.dtype)], dim=1)  # (B,2,3,H,W)
            prompts = tuple([""] * B)
            scores, _ = content_scorer(pair, prompts, None)
            return scores

        sim_model = None
    else:
        raise ValueError("similarity_backend must be 'lpips', 'face', or 'internvl-content'")
    if 'sim_model' in locals() and sim_model is not None:
        sim_model.requires_grad_(False)

    # ---------- photorealism model (optional) -----------------------
    photo_fn = None
    if sargs.photo_backend.lower() == "none":
        photo_fn = None
    elif sargs.photo_backend.lower() == "aide":
        try:
            from aide_scorer import make_aide_reward_fn
        except Exception as e:
            raise RuntimeError("aide_scorer.py not found or import failed") from e

        if sargs.aide_resume is None:
            raise RuntimeError("--aide_resume must be provided when photo_backend='aide'")

        photo_fn = make_aide_reward_fn(
            aide_repo=str(sargs.aide_repo),
            resume=str(sargs.aide_resume),
            resnet_path=None,
            convnext_path=None,
            device=device,
            dtype=dtype,
            token_size=256,
            calibrate_to_strict=False
        )
    else:
        raise ValueError("photo_backend must be 'none' or 'aide'")

    # ---------- diffusion pipeline ---------------------------------
    pipe = DDPOFluxKontextPipeline(
        sargs.pretrained_model,
        revision=sargs.pretrained_revision,
        use_lora=sargs.use_lora,
        lora_rank=sargs.lora_rank,
    ).to(device)

    # ---------- sharpness model (optional) --------------------------
    sharpness_fn = None
    if sargs.sharpness_backend.lower() == "internvl":
        internvl_model_id = "OpenGVLab/InternVL3-14B-hf"
        try:
            shared_processor  
            shared_model      
        except NameError:
            from internvl_scorer import load_internvl_core
            shared_processor, shared_model = load_internvl_core(
                internvl_model_id, device=device, dtype=dtype
            )
        from internvl_scorer import make_reward_fn as make_internvl_reward_fn
        sharp_scorer = make_internvl_reward_fn(
            model_id=internvl_model_id,
            task="sharpness",
            device=device,
            dtype=dtype,
            temperature=1.0,
            objective="expected",
            calibration="pmi_null",
            margin_temp=1.0,
            entropy_coef=0.0,
            processor=shared_processor,
            model=shared_model,
        )

        def sharpness_fn(images, *_):
            if images.dim() != 4 or images.size(1) != 3:
                raise ValueError("Expect images of shape (B,3,H,W)")
            B = images.size(0)
            images_batched = images.unsqueeze(1)
            prompts = tuple([""] * B)
            scores, aux = sharp_scorer(images_batched, prompts, None)
            return scores, aux

    # Configure grid penalty parameters
    grid_config = {
        "weight": sargs.grid_ratio_weight,
        "period": sargs.grid_ratio_period,
        "max_harm": sargs.grid_ratio_max_harm,
        "sigma_bins": sargs.grid_ratio_sigma_bins,
        "tau": sargs.grid_ratio_tau,
        "eval_resize": sargs.grid_eval_resize,
    }
    
    # Configure brightness/contrast anchors
    brightness_config = {
        "target": sargs.brightness_target,
        "band": sargs.brightness_band,
        "weight": sargs.brightness_anchor_weight,
        "crisp_weight": sargs.crispness_anchor_weight,
    }

    reward_fn = partial(
        combined_reward,
        aest_fn=aesthetic_fn,
        sim_fn=sim_fn,
        scale_aest=sargs.aesthetic_reward_scale,
        scale_img=sargs.image_reward_scale,
        photo_fn=photo_fn,
        scale_photo=sargs.photo_reward_scale,
        sharpness_fn=sharpness_fn,
        scale_sharpness=sargs.sharpness_reward_scale,
        grid_config=grid_config,
        brightness_config=brightness_config,
        log_debug=sargs.debug_images,
    )

    # Logging hook for training images
    def img_logger(batch, step, accel_logger):
        gen = batch["images"].to(dtype=torch.float32)
        prompts_b = batch["prompts"]
        refs = batch.get("input_images")
        if refs is not None:
            refs = refs.to(dtype=torch.float32)
            
        to_log = {}
        for i in range(min(4, len(gen))):
            prompt_key = prompts_b[i][:60].replace("/", "_")
            to_log[f"generated_{i}_{prompt_key}"] = gen[i : i + 1]
            if refs is not None:
                to_log[f"reference_{i}_{prompt_key}"] = refs[i : i + 1]
        
        accel_logger.log_images(to_log, step=step)

    # ---------- trainer --------------------------------------------
    trainer = AlignPropTrainerFluxKontext(
        train_cfg,
        reward_fn,
        prompt_fn,
        pipe,
        image_samples_hook=img_logger,
    )
    trainer.train()
    trainer.save_model(train_cfg.output_dir)
    if train_cfg.push_to_hub:
        trainer.push_to_hub(dataset_name="alignprop-finetuned-flux-kontext")


if __name__ == "__main__":
    main()
