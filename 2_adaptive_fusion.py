#!/usr/bin/env python3
"""Adaptive fusion: blend diffusion upscale with bicubic base using variance-weighted mask.
Low-variance regions (flat areas) trust the base more to suppress hallucinations."""
import argparse
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import uniform_filter


@dataclass
class SampleRegion:
    x: int
    y: int
    w: int
    h: int


def parse_sample(sample: str | None) -> SampleRegion | None:
    if not sample:
        return None
    vals = [int(v.strip()) for v in sample.split(",")]
    if len(vals) != 4:
        raise ValueError("sample must be x,y,w,h")
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        raise ValueError("sample w,h must be > 0")
    return SampleRegion(x=x, y=y, w=w, h=h)


def local_variance(gray: np.ndarray, window: int) -> np.ndarray:
    """Local variance map via box filter (mean and mean²)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window % 2 == 0:
        window += 1
    gray_f = gray.astype(np.float32)
    mean = uniform_filter(gray_f, size=window, mode="reflect")
    mean_sq = uniform_filter(gray_f * gray_f, size=window, mode="reflect")
    var = np.clip(mean_sq - mean * mean, 0, None)
    return var


def detect_scale(
    original_size: tuple[int, int],
    upscaled_size: tuple[int, int],
    sample: SampleRegion | None,
) -> tuple[float, float, str]:
    ow, oh = original_size
    uw, uh = upscaled_size
    scale_x = uw / ow
    scale_y = uh / oh

    if sample is None:
        return scale_x, scale_y, "full"

    # Check if upscaled is full-frame or sample crop
    expected_full_w = int(round(ow * scale_x))
    expected_full_h = int(round(oh * scale_y))
    if abs(expected_full_w - uw) <= 2 and abs(expected_full_h - uh) <= 2:
        return scale_x, scale_y, "full"

    scale_sample_x = uw / sample.w
    scale_sample_y = uh / sample.h
    return scale_sample_x, scale_sample_y, "sample_crop"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Path to original image.")
    parser.add_argument("--upscaled", required=True, help="Path to diffusion upscaled image.")
    parser.add_argument("--output", required=True, help="Path to write fused image.")
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Sample region in original coords: x,y,w,h",
    )
    parser.add_argument("--window", type=int, default=11, help="Variance window size.")
    parser.add_argument("--var-p-low", type=float, default=35.0, help="Lower percentile -> trust base more.")
    parser.add_argument("--var-p-high", type=float, default=85.0, help="Upper percentile -> trust diffusion more.")
    parser.add_argument("--min-detail-weight", type=float, default=0.15, help="Min diffusion weight in flat regions.")
    parser.add_argument("--max-detail-weight", type=float, default=0.85, help="Max diffusion weight in textured regions.")
    parser.add_argument("--blur-radius", type=float, default=4.0, help="Gaussian blur on weight mask.")
    args = parser.parse_args()

    if not (0 <= args.min_detail_weight <= 1 and 0 <= args.max_detail_weight <= 1):
        raise ValueError("min-detail-weight and max-detail-weight must be in [0,1]")
    if args.min_detail_weight > args.max_detail_weight:
        raise ValueError("min-detail-weight cannot exceed max-detail-weight")
    if args.var_p_low >= args.var_p_high:
        raise ValueError("var-p-low must be < var-p-high")

    sample = parse_sample(args.sample)

    original_path = os.path.join("input_images", args.original)
    upscaled_path = os.path.join("output_images", args.upscaled)
    
    original = Image.open(original_path).convert("RGB")
    upscaled = Image.open(upscaled_path).convert("RGB")
    ow, oh = original.size
    uw, uh = upscaled.size

    scale_x, scale_y, mode = detect_scale(original.size, upscaled.size, sample)
    print(f"Detected scale: x={scale_x:.4f}, y={scale_y:.4f}, mode={mode}")

    if sample is None:
        base = original.resize((uw, uh), resample=Image.Resampling.BICUBIC)
        orig_for_var = original
        up_work = upscaled
        out_canvas = None
    else:
        if mode == "sample_crop":
            x0 = max(0, sample.x)
            y0 = max(0, sample.y)
            x1 = min(ow, sample.x + sample.w)
            y1 = min(oh, sample.y + sample.h)
            orig_crop = original.crop((x0, y0, x1, y1))
            base = orig_crop.resize((uw, uh), resample=Image.Resampling.BICUBIC)
            orig_for_var = orig_crop
            up_work = upscaled
            out_canvas = None
        else:
            out_canvas = upscaled.copy()
            rx0 = int(round(sample.x * scale_x))
            ry0 = int(round(sample.y * scale_y))
            rw = int(round(sample.w * scale_x))
            rh = int(round(sample.h * scale_y))
            rx0 = max(0, min(uw, rx0))
            ry0 = max(0, min(uh, ry0))
            rx1 = max(rx0, min(uw, rx0 + rw))
            ry1 = max(ry0, min(uh, ry0 + rh))
            up_work = upscaled.crop((rx0, ry0, rx1, ry1))
            x0 = max(0, sample.x)
            y0 = max(0, sample.y)
            x1 = min(ow, sample.x + sample.w)
            y1 = min(oh, sample.y + sample.h)
            orig_crop = original.crop((x0, y0, x1, y1))
            base = orig_crop.resize(up_work.size, resample=Image.Resampling.BICUBIC)
            orig_for_var = orig_crop

    gray = np.array(orig_for_var.convert("L"), dtype=np.float32)
    var_map = local_variance(gray, args.window)
    if var_map.shape[::-1] != up_work.size:
        var_img = Image.fromarray(var_map.astype(np.float32), mode="F").resize(up_work.size, resample=Image.Resampling.BILINEAR)
        var_map = np.array(var_img, dtype=np.float32)

    v0 = float(np.percentile(var_map, args.var_p_low))
    v1 = float(np.percentile(var_map, args.var_p_high))
    denom = max(v1 - v0, 1e-6)
    w = np.clip((var_map - v0) / denom, 0.0, 1.0)
    w = args.min_detail_weight + w * (args.max_detail_weight - args.min_detail_weight)

    if args.blur_radius > 0:
        w_img = Image.fromarray((w * 255.0).astype(np.uint8), mode="L")
        w_img = w_img.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
        w = np.array(w_img, dtype=np.float32) / 255.0

    d = np.array(up_work, dtype=np.float32)
    b = np.array(base, dtype=np.float32)
    w3 = np.repeat(w[:, :, np.newaxis], 3, axis=2)
    fused = (w3 * d + (1.0 - w3) * b).clip(0, 255).astype(np.uint8)
    fused_img = Image.fromarray(fused)

    if out_canvas is None:
        result = fused_img
    else:
        rx0 = int(round(sample.x * scale_x))
        ry0 = int(round(sample.y * scale_y))
        out_canvas.paste(fused_img, (max(0, rx0), max(0, ry0)))
        result = out_canvas

    output_path = os.path.join("output_images", args.output)
    os.makedirs("output_images", exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path} ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    main()
