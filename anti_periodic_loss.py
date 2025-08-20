#!/usr/bin/env python3
"""
Anti-periodic loss functions for detecting and penalizing grid artifacts.

Provides differentiable losses to detect 16x16 periodic patterns and other 
grid artifacts in generated images. Includes both global FFT-based and 
local windowed approaches with edge-aware gating.
"""

import torch
import torch.nn.functional as F
import math

# Color space utilities
def srgb_to_linear(x):  # x in [0,1]
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a)/(1+a)).pow(2.4))

def linear_to_luminance(rgb):  # Rec.709 Y
    r, g, b = rgb.unbind(dim=-3)  # expects shape [..., 3, H, W]
    return 0.2126*r + 0.7152*g + 0.0722*b

def log_luminance(rgb, eps=1e-4):
    # rgb: [..., 3, H, W] in [0,1] sRGB
    lin = srgb_to_linear(rgb.clamp(0,1))
    Y = linear_to_luminance(lin)
    return (Y + eps).log()

# Frequency mask construction
def _gaussian_mask_2d(H, W, cy, cx, sy, sx, device):
    yy = torch.fft.fftfreq(H, device=device).view(H, 1).expand(H, W)
    xx = torch.fft.fftfreq(W, device=device).view(1, W).expand(H, W)
    return torch.exp(-0.5*(((yy-cy)/sy)**2 + ((xx-cx)/sx)**2))

def build_comb_mask_from_period(H, W, period=16, max_harm=6, sigma_bins=2.0,
                                vh_weight=(1.0, 1.0), device="cpu"):
    """
    Returns a fixed mask M(H,W) with Gaussian bumps at (m/period, n/period).
    sigma_bins is the std-dev measured in FFT "bins".
    """
    sy = sigma_bins / H
    sx = sigma_bins / W
    M = torch.zeros((H, W), device=device)
    for my in range(-max_harm, max_harm+1):
        for mx in range(-max_harm, max_harm+1):
            if my == 0 and mx == 0:
                continue
            w = 1.0
            if mx == 0 and my != 0: w = vh_weight[0]   # vertical family
            if my == 0 and mx != 0: w = vh_weight[1]   # horizontal family
            M = torch.maximum(M, w * _gaussian_mask_2d(H, W, my/period, mx/period, sy, sx, device))
    M[..., 0, 0] = 0.0
    return M  # (H,W)

def build_comb_mask_from_template(T16, H, W, period=16, topk=12, sigma_bins=2.0, device="cpu"):
    """
    Tailor the comb to YOUR extracted 16×16 tile:
    1) find the strongest frequency lines in FFT(T16),
    2) place Gaussians at those (m/period, n/period) across the full image grid.
    """
    t = torch.as_tensor(T16, dtype=torch.float32, device=device)  # 16×16
    t = t - t.mean()
    Ft = torch.fft.fft2(t)
    mag = Ft.abs()
    # keep top-k non-DC bins
    mag[0,0] = 0
    vals, idx = torch.topk(mag.reshape(-1), k=min(topk, mag.numel()-1))
    sy = sigma_bins / H
    sx = sigma_bins / W
    M = torch.zeros((H,W), device=device)
    for idv in idx.tolist():
        uy = idv // mag.shape[1]
        ux = idv %  mag.shape[1]
        # map 16×16 FFT bin (uy,ux) -> fundamental (my,mx) indices (signed) in units of 1/period
        my = (uy if uy <= 8 else uy-16)   # signed index in [-8,8]
        mx = (ux if ux <= 8 else ux-16)
        if my == 0 and mx == 0: continue
        M = torch.maximum(M, _gaussian_mask_2d(H, W, my/period, mx/period, sy, sx, device))
    M[...,0,0] = 0.0
    return M

# Global FFT-based losses
@torch.no_grad()
def _hann2d(H, W, device):
    wy = torch.hann_window(H, periodic=True, device=device)
    wx = torch.hann_window(W, periodic=True, device=device)
    return wy[:,None] * wx[None,:]

def comb_fft_loss(image_srgb, period=16, mask=None, max_harm=6, sigma_bins=2.0,
                  vh_weight=(1.0,1.0), use_hann=True, reduction="mean", tau=None):
    """
    Penalize energy at 16×16 comb frequencies (global).
    - image_srgb: [B,3,H,W] in [0,1]
    - mask: optional prebuilt M(H,W), else built from period.
    """
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)  # [B,H,W]

    if use_hann:
        w = _hann2d(H, W, device)
        Ylog = Ylog * w

    FY = torch.fft.fft2(Ylog)
    if mask is None:
        M = build_comb_mask_from_period(H, W, period, max_harm, sigma_bins, vh_weight, device)
    else:
        M = mask.to(device)
    # energy in masked bands
    E = (FY.abs()**2 * M).sum(dim=(-1,-2))  # [B]
    if tau is not None:
        E = F.relu(E - float(tau))
    if reduction == "mean":  return E.mean()
    if reduction == "sum":   return E.sum()
    return E

# Local windowed losses
def _unfold2d(x, win, hop):
    # x: [B,H,W] -> [B, n_patches, win, win]
    B, H, W = x.shape
    x4 = x[:,None,:,:]  # [B,1,H,W]
    patches = F.unfold(x4, kernel_size=(win,win), stride=(hop,hop), padding=win//2, dilation=1)  # [B, win*win, L]
    L = patches.shape[-1]
    patches = patches.transpose(1,2).reshape(B, L, 1, win, win)  # [B,L,1,win,win]
    return patches  # and we keep pad implicit (Fold not needed for loss)

def comb_local_loss(image_srgb, period=16, max_harm=6, sigma_bins=2.0,
                    vh_weight=(1.0,1.0), window=128, hop=32, reduction="mean", tau=None):
    """
    Local comb loss (no gating): average masked FFT energy per window.
    """
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)  # [B,H,W]
    patches = _unfold2d(Ylog, window, hop)  # [B,L,1,win,win]
    B, L, _, win, _ = patches.shape
    w = _hann2d(win, win, device)
    patches = (patches.squeeze(2) * w)  # [B,L,win,win]
    Fp = torch.fft.fft2(patches)        # [B,L,win,win]
    M = build_comb_mask_from_period(win, win, period, max_harm, sigma_bins, vh_weight, device)  # [win,win]
    E = (Fp.abs()**2 * M).sum(dim=(-1,-2))  # [B,L]
    E = E.mean(dim=1)                        # [B]
    if tau is not None:
        E = F.relu(E - float(tau))
    return E.mean() if reduction=="mean" else E.sum() if reduction=="sum" else E

# Edge-aware adaptive losses
def comb_local_adaptive_loss(image_srgb, period=16, max_harm=6, sigma_bins=2.0,
                             vh_weight=(1.0,1.0), window=128, hop=32,
                             edge_penalty=8.0, reduction="mean", tau=None):
    """
    Downweights subtraction near strong edges/speculars: gate = rel / (1 + edge_penalty * grad).
    """
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)  # [B,H,W]
    patches = _unfold2d(Ylog, window, hop)  # [B,L,1,win,win]
    B, L, _, win, _ = patches.shape
    w = _hann2d(win, win, device)
    P = patches.squeeze(2)                        # [B,L,win,win]
    Pw = P * w
    Fp = torch.fft.fft2(Pw)
    M = build_comb_mask_from_period(win, win, period, max_harm, sigma_bins, vh_weight, device)
    # comb-band energy ratio (reliability)
    Eh = (Fp.abs()**2 * M).sum(dim=(-1,-2))        # [B,L]
    Et = (Fp.abs()**2).sum(dim=(-1,-2)) + 1e-8     # [B,L]
    rel = Eh / Et                                  # [B,L]
    # gradient penalty (approximate)
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=device, dtype=P.dtype)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=device, dtype=P.dtype)
    gy = F.conv2d(P.view(B*L,1,win,win), kx, padding=1).abs()
    gx = F.conv2d(P.view(B*L,1,win,win), ky, padding=1).abs()
    gmean = (gy+gx).mean(dim=(-1,-2,-3)).view(B,L) # [B,L]
    gate = (rel / (1.0 + edge_penalty * gmean)).clamp(min=0.0, max=1.0) * 2.0
    gate = gate.clamp(0.0, 1.0)                    # [B,L]

    E = Eh * gate                                  # [B,L]
    E = E.mean(dim=1)                              # [B]
    if tau is not None:
        E = F.relu(E - float(tau))
    return E.mean() if reduction=="mean" else E.sum() if reduction=="sum" else E

# Template matching losses
def template_matched_loss(image_srgb, T16, strength=1.0, period=16, reduction="mean", tau=None):
    """
    Penalizes correlation with your extracted 16×16 template (phase-agnostic-ish).
    Implementation: conv2d with the normalized, zero-mean template; take L2 of response.
    For stronger phase-invariance, you can also add a bank of circular shifts of T16 (e.g., 4×4 phases).
    """
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)  # [B,H,W]

    T = torch.as_tensor(T16, dtype=torch.float32, device=device)
    T = T - T.mean()
    T = T / (T.norm() + 1e-8)
    k = T.view(1,1,period,period)

    # "same" convolution
    r = F.conv2d(Ylog.unsqueeze(1), k, padding=period//2)  # [B,1,H,W]
    L = (r**2).mean(dim=(-1,-2,-3)) * strength
    if tau is not None:
        L = F.relu(L - float(tau))
    return L.mean() if reduction=="mean" else L.sum() if reduction=="sum" else L

def _zero_mean(Y):
    """Subtract per-image spatial mean to remove DC component."""
    return Y - Y.mean(dim=(-2, -1), keepdim=True)


def _freq_grids(H, W, device):
    """Return frequency coordinate grids (cycles/pixel) after fftshift center.

    uy, ux have shape (H, W) with DC at (H//2, W//2); r is radial frequency.
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    uy = (yy - (H // 2)) / float(H)
    ux = (xx - (W // 2)) / float(W)
    r = torch.sqrt(uy * uy + ux * ux)
    return uy, ux, r


def comb_guard_ratio_dc_safe(
    image_srgb,
    *,
    period: int = 16,
    max_harm: int = 6,
    sigma_bins: float = 2.0,
    guard_mul: float = 3.0,
    lowfreq_cut: float = 0.02,
    use_hann: bool = True,
    reduction: str = "mean",
    tau: float | None = None,
):
    """
    DC-safe grid penalty using a guard-ring denominator.

    ratio = E_grid / E_guard, where E_guard is a widened ring around each comb line.
    Very low frequencies are excluded from the denominator to prevent brightness hacks.
    """
    x = image_srgb
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1.0) / 2.0

    # log luminance -> (optional) Hann -> zero-mean (remove DC)
    Ylog = log_luminance(x)
    B, H, W = Ylog.shape
    device = Ylog.device
    if use_hann:
        Ylog = Ylog * _hann2d(H, W, device)
    Ylog = _zero_mean(Ylog)

    Fy = torch.fft.fftshift(torch.fft.fft2(Ylog), dim=(-2, -1))
    P = (Fy.abs() ** 2)

    uy, ux, r = _freq_grids(H, W, device)
    sy = sigma_bins / H
    sx = sigma_bins / W

    f0 = 1.0 / float(period)
    grid_mask = torch.zeros((H, W), device=device)
    guard_mask = torch.zeros((H, W), device=device)

    def gline(val, center, hw):
        return torch.exp(-((val - center) ** 2) / (2.0 * (hw ** 2)))

    for k in range(1, int(max_harm) + 1):
        fk = k * f0
        if fk >= 0.5:
            break
        m  = gline(ux,  fk, sx) + gline(ux, -fk, sx) + gline(uy,  fk, sy) + gline(uy, -fk, sy)
        mg = gline(ux,  fk, guard_mul * sx) + gline(ux, -fk, guard_mul * sx) \
           + gline(uy,  fk, guard_mul * sy) + gline(uy, -fk, guard_mul * sy)
        grid_mask  = torch.maximum(grid_mask,  m)
        guard_mask = torch.maximum(guard_mask, mg)

    ring = (guard_mask - grid_mask).clamp_min(0.0)
    low = (r < float(lowfreq_cut)).to(P.dtype)
    ring = ring * (1.0 - low)

    E_grid  = (P * grid_mask).sum(dim=(-2, -1))
    E_guard = (P * ring).sum(dim=(-2, -1)) + 1e-12
    ratio   = E_grid / E_guard

    if tau is not None:
        ratio = F.relu(ratio - float(tau))
    if reduction == "mean":
        return ratio.mean()
    if reduction == "sum":
        return ratio.sum()
    return ratio


def noncomb_hf_floor(
    image_srgb,
    *,
    rmin: float = 0.06,
    rmax: float = 0.25,
    period: int = 16,
    max_harm: int = 6,
    sigma_bins: float = 2.0,
    guard_mul: float = 3.0,
    use_hann: bool = True,
    floor: float = 1e-3,
    weight: float = 0.25,
    reduction: str = "mean",
):
    """
    Encourage energy in a non-comb high-frequency band to avoid dim/blur collapse.
    Excludes comb lines from the safe band before measuring energy.
    """
    x = image_srgb
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1.0) / 2.0

    Ylog = log_luminance(x)
    B, H, W = Ylog.shape
    device = Ylog.device
    if use_hann:
        Ylog = Ylog * _hann2d(H, W, device)
    Ylog = _zero_mean(Ylog)

    Fy = torch.fft.fftshift(torch.fft.fft2(Ylog), dim=(-2, -1))
    P = (Fy.abs() ** 2)

    uy, ux, r = _freq_grids(H, W, device)
    safe = ((r >= float(rmin)) & (r <= float(rmax))).to(P.dtype)

    sy = sigma_bins / H
    sx = sigma_bins / W
    f0 = 1.0 / float(period)
    for k in range(1, int(max_harm) + 1):
        fk = k * f0
        if fk >= 0.5:
            break
        line = torch.exp(-((ux - fk) ** 2) / (2.0 * sx ** 2)) + torch.exp(-((ux + fk) ** 2) / (2.0 * sx ** 2)) \
             + torch.exp(-((uy - fk) ** 2) / (2.0 * sy ** 2)) + torch.exp(-((uy + fk) ** 2) / (2.0 * sy ** 2))
        safe = (safe * (1.0 - line.clamp_max(1.0))).clamp_min(0.0)

    E_safe = (P * safe).sum(dim=(-2, -1))
    loss = F.relu(float(floor) - E_safe) * float(weight)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def comb_fft_ratio_loss(image_srgb, period=16, mask=None, max_harm=6, sigma_bins=2.0,
                        vh_weight=(1.0,1.0), use_hann=True, reduction="mean", eps=1e-8, tau=None):
    """Global comb as ratio Eh/Et (contrast-normalized)."""
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)  # [B,H,W]
    if use_hann:
        Ylog = Ylog * _hann2d(H, W, device)
    FY = torch.fft.fft2(Ylog)
    if mask is None:
        M = build_comb_mask_from_period(H, W, period, max_harm, sigma_bins, vh_weight, device)
    else:
        M = mask.to(device)
    power = (FY.abs()**2)
    Eh = (power * M).sum(dim=(-1,-2))
    Et = power.sum(dim=(-1,-2)) + eps
    R = Eh / Et
    if tau is not None:
        R = F.relu(R - float(tau))
    return R.mean() if reduction=="mean" else R.sum() if reduction=="sum" else R


def comb_local_ratio_loss(image_srgb, period=16, max_harm=6, sigma_bins=2.0,
                          vh_weight=(1.0,1.0), window=128, hop=32, reduction="mean", eps=1e-8, tau=None):
    """Local overlap-add comb ratio averaged over windows."""
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)
    x4 = Ylog[:,None,:,:]
    P = F.unfold(x4, kernel_size=(window,window), stride=(hop,hop), padding=window//2)
    L = P.shape[-1]
    P = P.transpose(1,2).reshape(B, L, 1, window, window).squeeze(2)
    w = _hann2d(window, window, device)
    Pw = P * w
    Fp = torch.fft.fft2(Pw)
    M = build_comb_mask_from_period(window, window, period, max_harm, sigma_bins, vh_weight, device)
    power = (Fp.abs()**2)
    Eh = (power * M).sum(dim=(-1,-2))  # [B,L]
    Et = power.sum(dim=(-1,-2)) + eps
    R = (Eh / Et).mean(dim=1)          # [B]
    if tau is not None:
        R = F.relu(R - float(tau))
    return R.mean() if reduction=="mean" else R.sum() if reduction=="sum" else R


def comb_fft_ratio_from_template_loss(image_srgb, T16, period=16, topk=12, sigma_bins=1.2,
                                      reduction="mean", tau=None):
    """Global comb ratio using bins selected from a 16×16 template."""
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb)
    Ylog = Ylog * _hann2d(H, W, device)
    FY = torch.fft.fft2(Ylog)
    M = build_comb_mask_from_template(T16, H, W, period=period, topk=topk, sigma_bins=sigma_bins, device=device)
    power = (FY.abs()**2)
    Eh = (power * M).sum(dim=(-1,-2))
    Et = power.sum(dim=(-1,-2)) + 1e-8
    R = Eh / Et
    if tau is not None:
        R = F.relu(R - float(tau))
    return R.mean() if reduction=="mean" else R.sum() if reduction=="sum" else R


def phase_coherence_loss(image_srgb, period=16, max_harm=4, sigma_bins=1.0, vh_weight=(1.0,1.0),
                         reduction="mean", eps=1e-8, tau=None):
    """Phase-coherence of comb bins: |sum Z|^2 / (sum |Z|)^2."""
    B, C, H, W = image_srgb.shape
    device = image_srgb.device
    Ylog = log_luminance(image_srgb) * _hann2d(H, W, device)
    FY = torch.fft.fft2(Ylog)
    M = build_comb_mask_from_period(H, W, period, max_harm, sigma_bins, vh_weight, device)
    Z = FY * (M > 0).to(FY.dtype)
    num = (Z.sum(dim=(-1,-2)).abs()**2)
    den = (Z.abs().sum(dim=(-1,-2))**2) + eps
    coh = num / den
    if tau is not None:
        coh = F.relu(coh - float(tau))
    return coh.mean() if reduction=="mean" else coh.sum() if reduction=="sum" else coh


def folding_softmax_loss(image_srgb, period=16, temperature=0.5, reduction="mean", tau=None):
    """Differentiable shift-invariant folding criterion over period×period offsets."""
    B, C, H, W = image_srgb.shape
    Ylog = log_luminance(image_srgb)
    scores = []
    for dy in range(period):
        for dx in range(period):
            Yshift = torch.roll(Ylog, shifts=(dy, dx), dims=(-2,-1))
            T = F.avg_pool2d(Yshift.unsqueeze(1), kernel_size=period, stride=period).mean(dim=1)  # [B,h,w]
            s = (T - T.mean(dim=(-2,-1), keepdim=True)).pow(2).mean(dim=(-2,-1))  # [B]
            scores.append(s)
    S = torch.stack(scores, dim=-1)  # [B,period*period]
    w = torch.softmax(S / float(temperature), dim=-1)
    out = (S * w).sum(dim=-1)
    if tau is not None:
        out = F.relu(out - float(tau))
    return out.mean() if reduction=="mean" else out.sum() if reduction=="sum" else out


def _build_comb_masks(win, period=16, max_harm=6, sigma_bins=1.0, device="cpu"):
    """Return (notch_mask, guard_mask) for a windowed FFT grid comb.

    notch marks the comb lines; guard is a widened ring around them.
    """
    yy, xx = torch.meshgrid(
        torch.arange(win, device=device), torch.arange(win, device=device), indexing="ij"
    )
    uy = (yy - win // 2) / float(win)
    ux = (xx - win // 2) / float(win)
    sy = float(sigma_bins) / float(win)
    sx = float(sigma_bins) / float(win)
    f0 = 1.0 / float(period)
    notch = torch.zeros((win, win), device=device)
    guard = torch.zeros((win, win), device=device)

    def gline(val, c, h):
        return torch.exp(-((val - c) ** 2) / (2.0 * h * h))

    for k in range(1, int(max_harm) + 1):
        fk = k * f0
        if fk >= 0.5:
            break
        m = (
            gline(ux, fk, sx)
            + gline(ux, -fk, sx)
            + gline(uy, fk, sy)
            + gline(uy, -fk, sy)
        )
        mg = (
            gline(ux, fk, 3.0 * sx)
            + gline(ux, -fk, 3.0 * sx)
            + gline(uy, fk, 3.0 * sy)
            + gline(uy, -fk, 3.0 * sy)
        )
        notch = torch.maximum(notch, m)
        guard = torch.maximum(guard, mg)
    ring = (guard - notch).clamp_min(0.0)
    return notch, ring


def _sobel_grad_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean Sobel gradient magnitude per image.

    x: [N,1,H,W] → returns [N]
    """
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(x, kx, padding=1).abs()
    gy = F.conv2d(x, ky, padding=1).abs()
    return (gx + gy).mean(dim=(-1, -2, -3))


def selective_comb_local_loss(
    image_srgb,
    *,
    period: int = 16,
    max_harm: int = 6,
    window: int = 128,
    hop: int = 32,
    sigma_bins: float = 1.0,
    ratio_margin: float = 0.0,
    rel_margin: float = 0.002,
    edge_penalty: float = 10.0,
    top_p: float = 0.35,
    use_hann: bool = True,
    coverage_alpha: float = 0.5,
    return_parts: bool = False,
):
    """
    Local, selective, edge‑aware comb penalty (DC‑safe), using fftshift so masks align with centered coords.
    - ratio = E_grid / E_guard per window (grid/guard built in centered coordinates)
    - reliability = Eh / Et
    - edge gate downweights salient texture
    - penalty = mean( top‑p ratios ) * coverage_alpha
                + mean( ratios for all windows above both hinges ) * (1‑coverage_alpha)
    """
    x = image_srgb
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1) / 2

    # log‑luminance
    lin = srgb_to_linear(x.clamp(0, 1))
    Y = linear_to_luminance(lin).add_(1e-4).log()  # [B,H,W]
    B, H, W = Y.shape
    dev = Y.device

    # Unfold windows (overlap‑add)
    Y4 = Y[:, None, :, :]
    patches = F.unfold(Y4, kernel_size=window, stride=hop, padding=window // 2)
    L = patches.shape[-1]
    patches = patches.transpose(1, 2).reshape(B, L, 1, window, window)  # [B,L,1,win,win]

    # Hann + per‑patch zero‑mean (kills DC hack)
    if use_hann:
        w2d = _hann2d(window, window, dev)
        patches = patches * w2d
    patches = patches - patches.mean(dim=(-1, -2, -3), keepdim=True)

    # Centered spectrum to match masks
    Fp = torch.fft.fftshift(torch.fft.fft2(patches), dim=(-2, -1))
    P = (Fp.abs() ** 2).squeeze(2)  # [B,L,win,win]

    # Centered comb masks (notch & guard)
    notch, ring = _build_comb_masks(window, period=period, max_harm=max_harm, sigma_bins=sigma_bins, device=dev)

    Eh = (P * notch).sum(dim=(-1, -2))                 # [B,L]
    Eg = (P * ring).sum(dim=(-1, -2)) + 1e-12          # [B,L]
    Et = P.sum(dim=(-1, -2)) + 1e-12                   # [B,L]

    ratio = Eh / Eg                                    # grid/guard ratio (penalize high)
    rel = Eh / Et                                      # reliability (comb dominance)

    # Edge/saliency gate on raw (unwindowed) patches
    patches_raw = F.unfold(Y4, kernel_size=window, stride=hop, padding=window // 2)
    patches_raw = patches_raw.transpose(1, 2).reshape(B, L, 1, window, window)
    gmean = _sobel_grad_mean(patches_raw.reshape(B * L, 1, window, window)).view(B, L)
    edge_gate = torch.exp(-float(edge_penalty) * gmean).clamp(0.0, 1.0)

    # Hinge on ratio & reliability
    h_ratio = torch.relu(ratio - float(ratio_margin))
    h_rel = torch.relu(rel - float(rel_margin))
    weight = h_ratio * h_rel * edge_gate  # [B,L]

    # Coverage: top‑p and all‑above‑threshold
    k = max(1, int(round(L * float(top_p))))
    top_vals, top_idx = torch.topk(weight, k=k, dim=1, largest=True, sorted=False)
    top_ratio_mean = torch.gather(ratio, 1, top_idx).mean(dim=1)  # [B]

    mask_any = (weight > 0)
    any_count = mask_any.sum(dim=1).clamp_min(1)
    any_ratio_mean = (ratio * mask_any.to(ratio.dtype)).sum(dim=1) / any_count  # [B]

    loss_per_img = float(coverage_alpha) * top_ratio_mean + (1.0 - float(coverage_alpha)) * any_ratio_mean
    loss = loss_per_img.mean()

    if return_parts:
        return loss, {
            "ratio_top_mean": float(top_ratio_mean.mean().detach()),
            "ratio_any_mean": float(any_ratio_mean.mean().detach()),
            "Eh_mean": float(Eh.mean().detach()),
            "Eg_mean": float(Eg.mean().detach()),
            "Et_mean": float(Et.mean().detach()),
        }
    return loss

def comb_amplitude_hinge_local(
    image_srgb,
    *,
    period: int = 16,
    max_harm: int = 6,
    window: int = 128,
    hop: int = 32,
    sigma_bins: float = 1.0,
    amp_margin: float = 1e-3,
    use_hann: bool = True,
):
    """
    Gentle amplitude term that penalizes normalized comb energy above a small margin.
    Prevents gaming the ratio by inflating the guard.
    """
    x = image_srgb
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1) / 2

    lin = srgb_to_linear(x.clamp(0, 1))
    Y = linear_to_luminance(lin).add_(1e-4).log()
    B, H, W = Y.shape
    dev = Y.device

    Y4 = Y[:, None, :, :]
    Pch = F.unfold(Y4, kernel_size=window, stride=hop, padding=window // 2)
    L = Pch.shape[-1]
    Pch = Pch.transpose(1, 2).reshape(B, L, 1, window, window)
    if use_hann:
        w2d = _hann2d(window, window, dev)
        Pch = Pch * w2d
    Pch = Pch - Pch.mean(dim=(-1, -2, -3), keepdim=True)

    Fp = torch.fft.fftshift(torch.fft.fft2(Pch), dim=(-2, -1))
    Pwr = (Fp.abs() ** 2).squeeze(2)

    notch, _ = _build_comb_masks(window, period=period, max_harm=max_harm, sigma_bins=sigma_bins, device=dev)
    Eh = (Pwr * notch).sum(dim=(-1, -2))  # [B,L]
    Et = Pwr.sum(dim=(-1, -2)) + 1e-12
    amp = (Eh / Et).mean(dim=1)  # [B]
    return torch.relu(amp - float(amp_margin)).mean()

def _make_cosine_bases(win: int, period: int = 16, harmonics=(1, 2, 3), device: str | torch.device = "cpu"):
    """Build orthonormal 2D cosine/sine bases for k/period along x and y.
    Returns tensor of shape [K, win, win] with K = 4*len(harmonics).
    """
    y = torch.arange(win, device=device).view(win, 1).expand(win, win)
    x = torch.arange(win, device=device).view(1, win).expand(win, win)
    twopi = 2.0 * math.pi
    mats = []
    for k in harmonics:
        w = twopi * k / float(period)
        cos_y = torch.cos(w * y)
        sin_y = torch.sin(w * y)
        cos_x = torch.cos(w * x)
        sin_x = torch.sin(w * x)
        for M in (cos_y, sin_y, cos_x, sin_x):
            M = M - M.mean()
            M = M / (M.norm() + 1e-8)
            mats.append(M)
    return torch.stack(mats, dim=0)


def grid_cosine_projection_loss(
    image_srgb,
    *,
    period: int = 16,
    harmonics: tuple[int, ...] = (1, 2, 3),
    window: int = 128,
    hop: int = 32,
    use_hann: bool = True,
    edge_penalty: float = 12.0,
    top_p: float = 0.35,
    detach_norm: bool = True,
    return_parts: bool = False,
):
    """
    Phase‑invariant projection on period‑16 comb bases, windowed & edge‑aware.
    Penalizes only comb amplitude; optional detached normalization avoids exposure/contrast hack.
    Returns scalar loss; optionally returns diagnostics.
    """
    x = image_srgb
    if x.min() < 0:
        x = (x.clamp(-1, 1) + 1) / 2
    Y = linear_to_luminance(srgb_to_linear(x)).clamp_min(1e-6)
    B, H, W = Y.shape
    dev = Y.device

    Y4 = Y[:, None, :, :]
    P = F.unfold(Y4, kernel_size=window, stride=hop, padding=window // 2)
    L = P.shape[-1]
    P = P.transpose(1, 2).reshape(B, L, 1, window, window)

    if use_hann:
        w2d = _hann2d(window, window, dev)
        P = P * w2d
    P = P - P.mean(dim=(-1, -2, -3), keepdim=True)

    K = _make_cosine_bases(window, period=period, harmonics=harmonics, device=dev)  # [K,win,win]
    K = K.view(1, 1, K.size(0), window, window)

    coeffs = (P * K).sum(dim=(-1, -2))  # [B,L,K]
    Eh = (coeffs ** 2).sum(dim=-1)      # [B,L]

    En = (P ** 2).sum(dim=(-1, -2, -3)) + 1e-12  # [B,L]
    if detach_norm:
        En = En.detach()
    ratio = Eh / En  # [B,L]

    Praw = F.unfold(Y4, kernel_size=window, stride=hop, padding=window // 2)
    Praw = Praw.transpose(1, 2).reshape(B, L, 1, window, window)
    g = _sobel_grad_mean(Praw.reshape(B * L, 1, window, window)).view(B, L)
    edge_gate = torch.exp(-float(edge_penalty) * g).clamp(0.0, 1.0)

    weight = ratio * edge_gate
    k = max(1, int(round(L * float(top_p))))
    _, idx = torch.topk(weight, k=k, dim=1, largest=True, sorted=False)
    loss_per_img = torch.gather(ratio, 1, idx).mean(dim=1)
    loss = loss_per_img.mean()

    if return_parts:
        return loss, {
            "proj_ratio_top_mean": float(loss_per_img.mean().detach()),
            "proj_Eh_mean": float(Eh.mean().detach()),
            "proj_En_mean": float(En.mean().detach()),
        }
    return loss

def mean_luma_band(image_srgb, target: float = 0.50, band: float = 0.07, weight: float = 1.0, reduction: str = "mean"):
    """Zero penalty while mean luminance is within [target±band]; hinge outside.
    Returns scalar loss (not normalized by weight of other terms).
    """
    x = (image_srgb.clamp(-1, 1) + 1) / 2 if image_srgb.min() < 0 else image_srgb
    Y = linear_to_luminance(srgb_to_linear(x))
    m = Y.mean(dim=(-2, -1))
    low = float(target - band)
    high = float(target + band)
    pen = torch.relu(low - m) + torch.relu(m - high)
    pen = pen * float(weight)
    if reduction == "mean":
        return pen.mean()
    if reduction == "sum":
        return pen.sum()
    return pen


def tenengrad_floor_loss(image_srgb, floor: float = 1.5e-3, weight: float = 0.2):
    """Anti‑blur floor via Sobel energy (Tenengrad). Returns scalar hinge loss."""
    x = (image_srgb.clamp(-1, 1) + 1) / 2 if image_srgb.min() < 0 else image_srgb
    Y = linear_to_luminance(srgb_to_linear(x))
    kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=Y.device, dtype=Y.dtype)
    ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=Y.device, dtype=Y.dtype)
    gx = F.conv2d(Y[:, None], kx, padding=1)
    gy = F.conv2d(Y[:, None], ky, padding=1)
    eng = (gx * gx + gy * gy).mean(dim=(-1, -2, -3))
    return torch.relu(float(floor) - eng).mean() * float(weight)
