#!/usr/bin/env python3
"""
internvl_scorer.py — fully-differentiable reward model wrapper for InternVL

Supports tasks:
- single: rate one image on 0..9 (returns ~0..1, i.e., expected/9)
- artifact: rate overlay artifact/grid texture presence (0=heavy artifacts, 9=clean) -> is not working well
- pair: choose which image is better (returns probability for image 1)
- similarity: rate stylistic similarity of two images on 0..9 (returns ~0..1)
- sharpness: rate image sharpness/blur on 0..9 (0=very blurry, 9=very sharp)
 - triplet: pick which candidate (2 or 3) matches reference (1) in style (prob for candidate 1)
 - content-similarity: rate content/semantic similarity of two images on 0..9, ignoring style, lighting, background, and reflections

 Implements PMI-null calibration and anti-bias swapping for pair/triplet.
Gradients flow through image preprocessing and model forward.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Sequence
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText


GRADES = [str(i) for i in range(10)]   # tokens for rating tasks
PAIR_TOKENS = ["1", "2"]            # answers for pair / triplet


def _token_ids(text: str, tokenizer) -> List[int]:
    """Get token IDs for a text answer, trying common variants."""
    candidates, seen = [], set()
    for s in (text, " " + text, "\n" + text):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids and ids[-1] not in seen:
            candidates.append(ids[-1])
            seen.add(ids[-1])
    return candidates


def load_internvl_core(
    model_id: str = "OpenGVLab/InternVL3-14B-hf",
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load InternVL model and processor for reuse."""
    device = device if isinstance(device, torch.device) else torch.device(device)
    if device.type in ("cpu", "mps"):
        dtype = torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map=None
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return processor, model


def make_reward_fn(
    model_id: str = "OpenGVLab/InternVL3-14B-hf",
    *,
    task: str = "single",                 # single | artifact | pair | similarity | content-similarity | sharpness | triplet
    objective: str = "expected",          # expected | margin   (for single/similarity/content-similarity)
    temperature: float = 1.0,
    calibration: str = "pmi_null",        # pmi_null | none
    margin_temp: float = 1.0,
    entropy_coef: float = 0.0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    debug: bool = False,
    artifact_refs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    processor: Optional[AutoProcessor] = None,
    model: Optional[AutoModelForImageTextToText] = None,
    trace_layers: bool = False,            # when True, compute per-layer group logprobs (debug-only, heavy)
):
    """Create InternVL-based reward function.
    
    Returns scorer function that takes images and returns scores.
    """
    if task not in {"single", "artifact", "pair", "similarity", "sharpness", "triplet", "content-similarity"}:
        raise ValueError("task must be one of: single | artifact | pair | similarity | content-similarity | sharpness | triplet")
    if task in {"single", "artifact", "similarity", "sharpness", "content-similarity"} and objective not in {"expected", "margin"}:
        raise ValueError("objective must be expected|margin for task=single/similarity/content-similarity/sharpness/artifact")

    device = device if isinstance(device, torch.device) else torch.device(device)
    if device.type in ("cpu", "mps"):
        dtype = torch.float32

    # Model and processor setup
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_id)
    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map=None
        )
    # ensure device & eval
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Vision preprocessing parameters
    image_mean = torch.tensor(processor.image_processor.image_mean, device=device).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std, device=device).view(1, 3, 1, 1)
    # preferred spatial size from config
    vcfg = getattr(model.config, "vision_config", None)
    if vcfg is None:
        raise RuntimeError("InternVL model missing vision_config")
    image_size = vcfg.image_size
    if isinstance(image_size, int):
        target_h, target_w = image_size, image_size
    else:
        target_h, target_w = int(image_size[0]), int(image_size[1])

    param_dtype = next(model.parameters()).dtype
    # ensure normalization tensors match model dtype to avoid dtype promotion
    image_mean = image_mean.to(dtype=param_dtype)
    image_std = image_std.to(dtype=param_dtype)

    # Token mapping setup
    tok = processor.tokenizer
    grade_ids: Dict[str, List[int]] = {g: _token_ids(g, tok) for g in GRADES}
    pair_ids: Dict[str, List[int]] = {t: _token_ids(t, tok) for t in PAIR_TOKENS}
    scores_vec = torch.arange(10, device=device, dtype=torch.float32)

    # Prompt templates
    def prompt_single():
        return (
            "Please rate how professional and beautiful this image is. Is it a professional studio photograph with professional soft light done in a professional studio with a professional camera by a professional photographer? "
            "It should be a great high resolution photograph and it should not have any digital artifacts and should definitely not have a painterly texture. "
            "Rate from 0 (worst) to 9 (best) and answer with a single number."
        )

    def prompt_sharpness():
        return (
            "Please rate how sharp and in-focus this image is. Give a low score if it is blurry, or out of focus. "
            "Give a high score if fine details, edges, and textures appear crisp and well-resolved. Do not rate a smooth background or something that is unsharp due to shallow depth of field. Please ask yourself if the eyes of a portraited person are sharp, if there is skin texture, if the product on an product image is sharp etc. "
            "Are the eyes of a person sharp, do you see crisp skin texture and sharp hairs? Is on a product the texture sharp, the fabric sharp or the logos? "
            "Rate from 0 (very blurry) to 9 (very sharp) and answer with a single number. No explanation."
        )

    def prompt_artifact():
        if artifact_refs is not None:
            return (
                "There are exactly THREE images: Pictures 1 and 2 are reference examples illustrating overlay artifacts (grid/mesh/tiling, moiré, banding, or repeated dot/line textures). "
                "Picture 3 is the target image to evaluate. Rate the amount of such overlay artifacts present in Picture 3 from 0 (very heavy artifacts) to 9 (no artifacts). "
                "Answer with a single number: 0,1,2,3,4,5,6,7,8, or 9. No explanation."
            )
        else:
            return (
                "Please rate the amount of overlay artifacts such as grid/mesh/tiling, moiré, banding, or repeated texture patterns like small dots or lines that sit on top of the image. "
                "Give a low score if a visible grid or periodic overlay texture is present. Give a high score if the image is perfectly clean with no overlay artifacts. "
                "Rate from 0 (very heavy artifacts) to 9 (no artifacts at all) and answer with a single number."
            )

    def prompt_pair_choice():
        return (
            "There are two images. Please rate which image is more professional and more beautiful. Which one is more a professional studio photograph with professional soft light done in a professional studio with a professional camera by a professional photographer? "
            "The winner should be a great high resolution photograph and it should not have any digital artifacts and should definitely not have a painterly texture. "
            "Answer with '1' if the FIRST image is better, '2' if the SECOND is better. "
            "No explanation."
        )

    def prompt_similarity():
        return (
            "There are two images. Rate the stylistic similarity between the two images on a scale from 0 (very different) to 9 (very similar). "
            "Do not judge if the content of the images is similar. It is only about the style. Ask yourself if the two images could be from the same photoshoot, the same photographer or in the same magazine. "
            "How similar is the lighting, the background and the colorscheme? Pay extra attention to the colors (brightness, saturation, contrast). "
            "Answer with a single number: 0,1,2,3,4,5,6,7,8, or 9. No explanation."
        )

    def prompt_content_similarity():
        return (
            "There are two images. Rate the CONTENT similarity between the two images on a scale from 0 (very different) to 9 (very similar). "
            "Focus ONLY on what is depicted: the same person or objects their identities, clothes, fabric detail, spatial arrangement, and any text or logos. "
            "Completely IGNORE differences in lighting, color, background, reflections, camera quality, or stylistic post-processing.Only focus on the main subject and the main object."
            "Answer with a single number: 0,1,2,3,4,5,6,7,8, or 9. No explanation."
        )

    def prompt_triplet():
        return (
            "There are exactly THREE images: Picture 1 = REFERENCE, Picture 2 = CANDIDATE 1, Picture 3 = CANDIDATE 2. "
            "Pick the candidate whose STYLE is closer to the REFERENCE Picture 1. "
            "Do not judge if the content of the images is similar. It is only about the style. "
            "Answer with exactly one character: '1' if Picture 2 is closer, or '2' if Picture 3 is closer."
        )

    build_prompt = {
        "single": prompt_single,
        "artifact": prompt_artifact,
        "pair": prompt_pair_choice,
        "similarity": prompt_similarity,
        "sharpness": prompt_sharpness,
        "triplet": prompt_triplet,
        "content-similarity": prompt_content_similarity,
    }[task]

    # Image preprocessing
    def _preprocess_images(images: torch.Tensor) -> torch.Tensor:
        # images: (B,N,3,H,W) in [-1,1]
        B, N, C, H, W = images.shape
        x = images
        if x.min() < 0:
            x = (x.clamp(-1, 1) + 1.0) / 2.0  # to 0..1
        x = F.interpolate(
            x.view(B * N, C, H, W), size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        x = (x - image_mean) / image_std
        return x.to(device=device, dtype=param_dtype)

    # Batch construction
    def _build_batch(images: torch.Tensor, prompts: Sequence[str]):
        B, N, C, H, W = images.shape
        # If artifact references are enabled, prepend two reference images per sample
        if task == "artifact" and artifact_refs is not None:
            # Resize refs to current (H,W)
            refs_hw = torch.stack(artifact_refs, dim=0).to(device=device, dtype=dtype)  # (2,3,*,*)
            refs_hw = F.interpolate(refs_hw, size=(H, W), mode="bilinear", align_corners=False)
            refs_b = refs_hw.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B,2,3,H,W)
            images = torch.cat([refs_b, images], dim=1)
            N = N + 2
        # Construct chat messages with N image placeholders per sample
        messages = []
        images_grouped = []
        dummy = Image.new("RGB", (target_w, target_h))
        for i in range(B):
            content = [{"type": "image"} for _ in range(N)]
            content.append({"type": "text", "text": build_prompt()})
            messages.append([{"role": "user", "content": content}])
            images_grouped.append([dummy] * N)

        # Tokenize using processor; images are only used to create correct placeholders/masks
        with torch.no_grad():
            texts = [
                processor.apply_chat_template(m, add_generation_prompt=True, add_vision_id=True)
                for m in messages
            ]
            enc = processor(text=texts, images=images_grouped, padding=True, return_tensors="pt")

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Our differentiable pixel values
        pixel_values = _preprocess_images(images)  # (B*N,3,Ht,Wt)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "B": B,
            "N": N,
        }

    # Model forward pass
    def _get_attr(obj, dotted: str):
        cur = obj
        for part in dotted.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    lm_head =  _get_attr(model, "lm_head")

    def _forward(input_ids, attention_mask, pixel_values, need_hidden: bool = False):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cache=False,
            output_hidden_states=bool(need_hidden),
            return_dict=True,
        )
        logits = out.logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / float(temperature)
        logp_last = logits.float().log_softmax(dim=-1)

        layer_hidden = None
        if need_hidden:
            layer_hidden = getattr(out, "hidden_states", None)
        return logp_last, layer_hidden

    def _layer_group_logprobs(layer_hidden, input_ids, attention_mask, pixel_values):
        """Compute per-layer token group logprobs for debugging."""
        if layer_hidden is None or lm_head is None:
            return {}
        with torch.no_grad():
            layers = list(layer_hidden)
            B = layers[0].shape[0]
            grade_traces: List[torch.Tensor] = []
            pair_traces: List[torch.Tensor] = []
            for h in layers:
                # h: (B, T, C) → take last token position
                ht = h[:, -1, :]
                lg = lm_head(ht)  # (B, V)
                if temperature != 1.0:
                    lg = lg / float(temperature)
                lp = lg.float().log_softmax(dim=-1)
                if task in ("single", "artifact", "similarity", "sharpness", "content-similarity"):
                    cols = [torch.logsumexp(lp[:, ids], dim=-1) for ids in grade_ids.values()]
                    G = torch.stack(cols, dim=-1)  # (B,10) unnormalized log-scores per grade group
                    grade_traces.append(G)
                elif task in ("pair", "triplet"):
                    p1 = torch.logsumexp(lp[:, pair_ids["1"]], dim=-1)
                    p2 = torch.logsumexp(lp[:, pair_ids["2"]], dim=-1)
                    pair_traces.append(torch.stack([p1, p2], dim=-1))  # (B,2)
            out = {}
            if grade_traces:
                out["grades_raw_log"] = torch.stack(grade_traces, dim=1)  # (B,L,10)
            if pair_traces:
                out["pair_raw_log"] = torch.stack(pair_traces, dim=1)    # (B,L,2)
            return out

    # Utility functions
    def _logprob(tok_str: str, logp: torch.Tensor):
        ids = pair_ids[tok_str] if tok_str in pair_ids else grade_ids[tok_str]
        return torch.logsumexp(logp[:, ids], dim=-1)

    def _collapse(logp: torch.Tensor, mapping: Dict[str, List[int]]):
        cols = [torch.exp(torch.logsumexp(logp[:, ids], dim=-1)) for ids in mapping.values()]
        P = torch.stack(cols, dim=-1)
        return P / (P.sum(dim=-1, keepdim=True) + 1e-12)

    def _entropy(p: torch.Tensor):
        return -(p.clamp_min(1e-12).log() * p).sum(-1)

    def _pair_probs_from_logp(logp: torch.Tensor) -> torch.Tensor:
        """Get normalized probabilities for tokens '1' and '2'."""
        p1 = torch.exp(_logprob("1", logp))
        p2 = torch.exp(_logprob("2", logp))
        s = (p1 + p2).clamp_min(1e-12)
        return torch.stack([p1 / s, p2 / s], dim=-1)

    # Main scoring function
    @torch.enable_grad()
    def scorer(images: torch.Tensor, prompts: Tuple[str, ...], _meta=None):
        batch = _build_batch(images, prompts)
        need_trace = bool(trace_layers) or bool(_meta.get("trace_layers")) if isinstance(_meta, dict) else bool(trace_layers)
        logp_img, hs_img = _forward(batch["input_ids"], batch["attention_mask"], batch["pixel_values"], need_hidden=need_trace)

        if calibration == "pmi_null":
            with torch.no_grad():
                # gray in pre-normalized 0..1 → normalize (keep dtype consistent with model)
                null_pix = torch.full_like(batch["pixel_values"], 0.5)
                null_pix = (null_pix - image_mean) / image_std
                null_pix = null_pix.to(dtype=param_dtype)
            logp_null, hs_null = _forward(batch["input_ids"], batch["attention_mask"], null_pix, need_hidden=need_trace)
            logp = logp_img - logp_null
        elif calibration == "none":
            logp = logp_img
        else:
            raise ValueError(f"Unknown calibration: {calibration}")

        # optional: per-layer tracing
        trace_aux = {}
        if need_trace:
            trace_img = _layer_group_logprobs(hs_img, batch["input_ids"], batch["attention_mask"], batch["pixel_values"])
            if calibration == "pmi_null":
                trace_null = _layer_group_logprobs(hs_null, batch["input_ids"], batch["attention_mask"], null_pix)
                # subtract token-group log-scores layerwise (PMI per group)
                if "grades_raw_log" in trace_img and "grades_raw_log" in trace_null:
                    trace_aux["layer_grade_logprobs"] = trace_img["grades_raw_log"] - trace_null["grades_raw_log"]
                if "pair_raw_log" in trace_img and "pair_raw_log" in trace_null:
                    trace_aux["layer_pair_logprobs"] = trace_img["pair_raw_log"] - trace_null["pair_raw_log"]
            else:
                if "grades_raw_log" in trace_img:
                    trace_aux["layer_grade_logprobs"] = trace_img["grades_raw_log"]
                if "pair_raw_log" in trace_img:
                    trace_aux["layer_pair_logprobs"] = trace_img["pair_raw_log"]

        if task in ("single", "artifact", "sharpness"):
            P = _collapse(logp, grade_ids)  # (B,10)
            if objective == "expected":
                score = (P * scores_vec).sum(-1) / 9.0
            else:  # margin 9 vs 0
                margin = _logprob("9", logp) - _logprob("0", logp)
                score = torch.sigmoid(margin / max(1e-6, float(margin_temp)))
            if entropy_coef:
                score = score + float(entropy_coef) * _entropy(P)
            aux_out = {"grade_probs": P}
            aux_out.update(trace_aux)
            return score, aux_out

        if task in ("similarity", "content-similarity"):
            # Two images present; we still read the last-token distribution for a 0..9 answer
            P = _collapse(logp, grade_ids)  # (B,10)
            if objective == "expected":
                score = (P * scores_vec).sum(-1) / 9.0
            else:  # margin 9 vs 0
                margin = _logprob("9", logp) - _logprob("0", logp)
                score = torch.sigmoid(margin / max(1e-6, float(margin_temp)))
            if entropy_coef:
                score = score + float(entropy_coef) * _entropy(P)
            aux_out = {"grade_probs": P}
            aux_out.update(trace_aux)
            return score, aux_out

        if task == "pair":
            # original order [img1, img2]
            p1_a, p2_a = _logprob("1", logp), _logprob("2", logp)
            s_a = torch.sigmoid((p1_a - p2_a) / max(1e-6, float(margin_temp)))
            pair_probs_a = _pair_probs_from_logp(logp)
            # swapped order [img2, img1] — reuse inputs, only swap pixel order
            B, N = int(batch["B"]), int(batch["N"])  # N must be 2
            assert N == 2, "pair task expects N=2"
            pix_swapped = batch["pixel_values"].clone().view(B, N, 3, target_h, target_w)[:, [1, 0], ...].reshape(B * N, 3, target_h, target_w)
            logp_b, _ = _forward(batch["input_ids"], batch["attention_mask"], pix_swapped, need_hidden=False)
            p1_b, p2_b = _logprob("1", logp_b), _logprob("2", logp_b)
            s_b = torch.sigmoid((p1_b - p2_b) / max(1e-6, float(margin_temp)))
            score = 0.5 * (s_a + (1.0 - s_b))
            pair_probs_b = _pair_probs_from_logp(logp_b)
            # anti-symmetric combine in probability space
            prob1_combined = 0.5 * (pair_probs_a[:, 0] + (1.0 - pair_probs_b[:, 0]))
            pair_probs_combined = torch.stack([prob1_combined, 1.0 - prob1_combined], dim=-1)
            aux = {
                "margin_a": (p1_a - p2_a),
                "margin_b": (p1_b - p2_b),
                "pair_probs_a": pair_probs_a,
                "pair_probs_b": pair_probs_b,
                "pair_probs_combined": pair_probs_combined,
            }
            aux.update(trace_aux)
            return score, aux

        # triplet: compare candidates 1 vs 2 relative to reference 0
        B, N = int(batch["B"]), int(batch["N"])  # N must be 3
        assert N == 3, "triplet task expects N=3"
        p1_a, p2_a = _logprob("1", logp), _logprob("2", logp)
        s_a = torch.sigmoid((p1_a - p2_a) / max(1e-6, float(margin_temp)))
        pair_probs_a = _pair_probs_from_logp(logp)
        # swapped candidates: [ref, cand2, cand1]
        pix_swapped = batch["pixel_values"].clone().view(B, N, 3, target_h, target_w)[:, [0, 2, 1], ...].reshape(B * N, 3, target_h, target_w)
        logp_b, _ = _forward(batch["input_ids"], batch["attention_mask"], pix_swapped, need_hidden=False)
        p1_b, p2_b = _logprob("1", logp_b), _logprob("2", logp_b)
        s_b = torch.sigmoid((p1_b - p2_b) / max(1e-6, float(margin_temp)))
        score = 0.5 * (s_a + (1.0 - s_b))
        pair_probs_b = _pair_probs_from_logp(logp_b)
        prob1_combined = 0.5 * (pair_probs_a[:, 0] + (1.0 - pair_probs_b[:, 0]))
        pair_probs_combined = torch.stack([prob1_combined, 1.0 - prob1_combined], dim=-1)
        aux = {
            "margin_a": (p1_a - p2_a),
            "margin_b": (p1_b - p2_b),
            "pair_probs_a": pair_probs_a,
            "pair_probs_b": pair_probs_b,
            "pair_probs_combined": pair_probs_combined,
        }
        aux.update(trace_aux)
        return score, aux

    return scorer


# -------------------- CLI helpers --------------------
def load_img(path: Path, H=1024, W=512, device="cpu", dtype=torch.float32, requires_grad=False):
    try:
        img = Image.open(path).convert("RGB").resize((W, H))
        arr = torch.from_numpy(np.asarray(img).astype("float32") / 255.0).permute(2, 0, 1)
    except Exception as e:
        print(f"[warn] failed to load image {path}: {e}; using gray fallback")
        arr = torch.full((3, H, W), 0.5, dtype=torch.float32)
    t = (arr * 2 - 1).to(device=device, dtype=dtype).requires_grad_(requires_grad)
    return t


def group_paths(paths: List[Path], task: str) -> List[List[Path]]:
    need = {"single": 1, "artifact": 1, "sharpness": 1, "pair": 2, "similarity": 2, "content-similarity": 2, "triplet": 3}[task]
    if len(paths) % need != 0:
        raise RuntimeError(f"{task} needs {need}*k images (got {len(paths)}).")
    return [paths[i : i + need] for i in range(0, len(paths), need)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--task", choices=["single", "pair", "similarity", "content-similarity", "triplet", "artifact", "sharpness"], default="single")
    ap.add_argument("--objective", choices=["expected", "margin"], default="expected")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--model_id", default="OpenGVLab/InternVL3-14B-hf")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--margin_temp", type=float, default=1.0)
    ap.add_argument("--entropy_coef", type=float, default=0.0)
    ap.add_argument("--calibration", choices=["pmi_null", "none"], default="none")
    ap.add_argument("--print_probs", action="store_true", help="Print per-token probabilities")
    ap.add_argument("--trace_layers", action="store_true", help="Enable per-layer token-group logprob tracing (debug)")
    ap.add_argument("--print_layer_logprobs", action="store_true", help="Print per-layer logprob traces (can be very verbose)")
    ap.add_argument("--print_layer_probs", action="store_true", help="Print per-layer probabilities (normalized like final probs), with Δmean to final")
    ap.add_argument("--artifact_ref1", type=str, default=None, help="Path to reference image 1 (artifact example)")
    ap.add_argument("--artifact_ref2", type=str, default=None, help="Path to reference image 2 (artifact example)")
    ap.add_argument("--viz_grads", action="store_true", help="Compute and save gradient visualizations")
    ap.add_argument("--viz_outdir", default="debug_images", help="Directory to save gradient visualizations")
    ap.add_argument("--viz_alpha", type=float, default=0.6, help="Overlay strength (0..1)")
    ap.add_argument(
        "--viz_norm",
        choices=["percentile", "log", "none"],
        default="percentile",
        help="Dynamic range compression for gradients",
    )
    ap.add_argument("--viz_pmin", type=float, default=2.0, help="Lower percentile for percentile norm (0..100)")
    ap.add_argument("--viz_pmax", type=float, default=98.0, help="Upper percentile for percentile norm (0..100)")
    ap.add_argument("--viz_log_alpha", type=float, default=10.0, help="Alpha for log compression (higher=more compression)")
    ap.add_argument("--viz_blur", type=float, default=0.0, help="Gaussian blur sigma (px) for smoothing the heatmap")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    if device.type in ("cpu", "mps"):
        dtype = torch.float32

    # Load optional artifact references
    art_refs = None
    if args.task == "artifact" and args.artifact_ref1 and args.artifact_ref2:
        p1, p2 = Path(args.artifact_ref1), Path(args.artifact_ref2)
        if not p1.is_file() or not p2.is_file():
            print("[warn] artifact references not found; continuing without refs")
        else:
            # Load w/o grad; size will be adapted in batch
            r1 = load_img(p1, device=device, dtype=dtype, requires_grad=False)
            r2 = load_img(p2, device=device, dtype=dtype, requires_grad=False)
            art_refs = (r1, r2)

    scorer = make_reward_fn(
        model_id=args.model_id,
        task=args.task,
        objective=args.objective,
        temperature=args.temperature,
        margin_temp=args.margin_temp,
        entropy_coef=args.entropy_coef,
        calibration=args.calibration,
        device=device,
        dtype=dtype,
        artifact_refs=art_refs,
        trace_layers=bool(args.trace_layers),
    )

    # collect image paths
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    flat: List[Path] = []
    for inp in args.inputs:
        p = Path(inp).expanduser()
        if p.is_dir():
            flat += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in exts]
        elif p.is_file() and p.suffix.lower() in exts:
            flat.append(p)
    if not flat:
        sys.exit("no images found")

    groups = group_paths(flat, args.task)
    print(f"{len(groups)} samples | device={device} | task={args.task}")

    # Gradient visualization helper
    def _save_grad_images(images_bnchw: torch.Tensor, grads_bnchw: torch.Tensor, batch_paths: List[List[Path]]):
        import datetime as _dt
        from PIL import Image
        out_root = Path(args.viz_outdir) / ("gradviz_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        out_root.mkdir(parents=True, exist_ok=True)
        B, N = images_bnchw.shape[:2]
        imgs = images_bnchw.detach().cpu()
        grads = grads_bnchw.detach().cpu()

        def _gaussian_blur_hw(x: torch.Tensor, sigma: float) -> torch.Tensor:
            if sigma <= 0:
                return x
            # x: (H,W)
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
            # g_map: (H,W), non-negative
            if args.viz_blur > 0.0:
                g_map_blur = _gaussian_blur_hw(g_map, float(args.viz_blur))
            else:
                g_map_blur = g_map
            if args.viz_norm == "percentile":
                flat = g_map_blur.flatten()
                lo = torch.quantile(flat, float(args.viz_pmin) / 100.0)
                hi = torch.quantile(flat, float(args.viz_pmax) / 100.0)
                g = (g_map_blur - lo) / (hi - lo + 1e-12)
                return g.clamp(0, 1)
            elif args.viz_norm == "log":
                m = g_map_blur.max()
                g = g_map_blur / (m + 1e-12)
                g = torch.log1p(float(args.viz_log_alpha) * g) / np.log1p(float(args.viz_log_alpha))
                return g.clamp(0, 1)
            else:
                m = g_map_blur.max()
                return (g_map_blur / (m + 1e-12)).clamp(0, 1)
        for b in range(B):
            for n in range(N):
                img = imgs[b, n]
                g = grads[b, n]
                # to [0,1]
                rgb = ((img.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
                # gradient magnitude heatmap
                g_map = g.abs().mean(0)
                g_map = _compress(g_map)
                # overlay in red
                overlay = rgb.clone()
                overlay[0] = (1 - args.viz_alpha) * overlay[0] + args.viz_alpha * g_map
                overlay[1] = (1 - args.viz_alpha) * overlay[1]
                overlay[2] = (1 - args.viz_alpha) * overlay[2]
                # save
                def to_pil(t):
                    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype("uint8"))
                name_prefix = f"{b}_{n}_" + ",".join(p.name for p in batch_paths[b])
                to_pil(rgb).save(out_root / f"{name_prefix}_input.png")
                to_pil(g_map.expand_as(rgb)).save(out_root / f"{name_prefix}_grad.png")
                to_pil(overlay).save(out_root / f"{name_prefix}_overlay.png")

    for i in range(0, len(groups), args.batch_size):
        batch_paths = groups[i : i + args.batch_size]
        tensor = torch.stack(
            [
                torch.stack(
                    [
                        load_img(p, device=device, dtype=dtype, requires_grad=args.viz_grads)
                        for p in grp
                    ],
                    0,
                )
                for grp in batch_paths
            ],
            0,
        )  # (B,N,3,H,W)
        prompts = tuple(["" for _ in range(tensor.shape[0])])
        scores, aux = scorer(tensor, prompts, None)
        for b_idx, (grp, s) in enumerate(zip(batch_paths, scores)):
            line = f"{','.join(p.name for p in grp)}\t{s.item():.4f}"
            if args.print_probs:
                if args.task in ("single", "similarity", "content-similarity"):
                    probs = aux["grade_probs"][b_idx].detach().cpu().tolist()
                    probs_str = ",".join(f"{v:.4f}" for v in probs)
                    line += f"\tgrades:[{probs_str}]"
                elif args.task in ("pair", "triplet"):
                    pa = aux["pair_probs_a"][b_idx].detach().cpu().tolist()
                    pb = aux["pair_probs_b"][b_idx].detach().cpu().tolist()
                    pc = aux["pair_probs_combined"][b_idx].detach().cpu().tolist()
                    line += (
                        f"\tpair_a:[{pa[0]:.4f},{pa[1]:.4f}]"
                        f"\tpair_b:[{pb[0]:.4f},{pb[1]:.4f}]"
                        f"\tpair_comb:[{pc[0]:.4f},{pc[1]:.4f}]"
                    )
            print(line)
            if args.print_layer_logprobs:
                # Pretty-print per-layer tables
                def _print_table(title: str, mat, col_labels):
                    try:
                        import numpy as _np  # noqa
                        arr = mat
                        rows = int(arr.shape[0])
                        cols = int(arr.shape[1])
                    except Exception:
                        return
                    print(f"    {title}:")
                    header = "      L | " + " ".join(f"{lab:^8}" for lab in col_labels)
                    print(header)
                    print("      " + "-" * (len(header) - 6))
                    for li in range(rows):
                        vals = " ".join(f"{float(arr[li, cj]):8.3f}" for cj in range(cols))
                        print(f"     {li:>3} | {vals}")
                if "layer_grade_logprobs" in aux:
                    lg = aux["layer_grade_logprobs"][b_idx].detach().cpu().numpy()  # (L,10)
                    _print_table("layer_grade_logprobs (per token 0..9)", lg, [str(i) for i in range(10)])
                if "layer_pair_logprobs" in aux:
                    lp = aux["layer_pair_logprobs"][b_idx].detach().cpu().numpy()  # (L,2)
                    _print_table("layer_pair_logprobs (token '1','2')", lp, ["1", "2"])

            if args.print_layer_probs:
                # Pretty-print per-layer probabilities with Delta mean to final
                def _print_probs_table(title: str, mat_log, final_probs, col_labels):
                    try:
                        import numpy as _np  # noqa
                        lg = mat_log  # (L,C)
                        # softmax along classes
                        m = lg.max(axis=1, keepdims=True)
                        ex = _np.exp(lg - m)
                        pl = ex / (ex.sum(axis=1, keepdims=True) + 1e-12)
                        # Delta mean L1 to final
                        diff = _np.abs(pl - final_probs.reshape(1, -1)).mean(axis=1)
                        argm = pl.argmax(axis=1)
                        rows, cols = int(pl.shape[0]), int(pl.shape[1])
                    except Exception:
                        return
                    print(f"    {title}:")
                    header = "      L | " + " ".join(f"{lab:^8}" for lab in col_labels) + "  |  argmax  |  Δmean"
                    print(header)
                    print("      " + "-" * (len(header) - 6))
                    for li in range(rows):
                        vals = " ".join(f"{float(pl[li, cj]):8.3f}" for cj in range(cols))
                        print(f"     {li:>3} | {vals}  |  {int(argm[li]):>6}  |  {float(diff[li]):.3f}")
                if "layer_grade_logprobs" in aux and "grade_probs" in aux:
                    lg = aux["layer_grade_logprobs"][b_idx].detach().cpu().numpy()  # (L,10)
                    pf = aux["grade_probs"][b_idx].detach().cpu().numpy()          # (10,)
                    _print_probs_table("layer_grade_probs (per token 0..9)", lg, pf, [str(i) for i in range(10)])
                if "layer_pair_logprobs" in aux and "pair_probs_combined" in aux and args.task == "pair":
                    lp = aux["layer_pair_logprobs"][b_idx].detach().cpu().numpy()  # (L,2)
                    pf = aux["pair_probs_combined"][b_idx].detach().cpu().numpy()  # (2,)
                    _print_probs_table("layer_pair_probs (token '1','2')", lp, pf, ["1", "2"])
        if args.viz_grads:
            # get grads w.r.t inputs
            g = torch.autograd.grad(scores.sum(), tensor, retain_graph=False, allow_unused=True)[0]
            if g is None:
                print("[warn] gradient w.r.t. inputs is None; ensure inputs require_grad and graph not detached.")
            else:
                _save_grad_images(tensor, g, batch_paths)


if __name__ == "__main__":
    main()


