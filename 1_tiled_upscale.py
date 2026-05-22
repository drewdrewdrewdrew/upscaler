import argparse
import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
from tqdm.auto import tqdm


def tile_upscale(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    tile_size: int = 128,
    overlap: int = 32,
    step: int = None,
    noise_level: int = 10,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 50,
    seed: int = 42,
    sample_region: tuple = None,
    fidelity: float = 0.0,
):
    """
    Upscale an image 4x using tiled diffusion with blended overlaps.

    sample_region: (x, y, w, h) in input-space pixels. If set, only tiles
    overlapping this region are processed and the output is cropped to it.
    """
    w, h = image.size
    scale = 4
    out_tile = tile_size * scale
    step = step if step is not None else (tile_size - overlap)
    if step <= 0 or step > tile_size:
        raise ValueError(f"step must be in [1, tile_size], got {step} with tile_size={tile_size}")
    if not (0.0 <= fidelity <= 1.0):
        raise ValueError(f"fidelity must be in [0, 1], got {fidelity}")

    # Reflect-pad to improve edge context for tiles near region/image borders.
    img_arr = np.array(image)
    padded_arr = np.pad(img_arr, ((overlap, overlap), (overlap, overlap), (0, 0)), mode="reflect")
    padded = Image.fromarray(padded_arr)
    pw, ph = padded.size

    def get_positions(length, tile, step):
        positions = list(range(0, length - tile, step))
        if not positions or positions[-1] + tile < length:
            positions.append(max(0, length - tile))
        return positions

    x_positions = get_positions(pw, tile_size, step)
    y_positions = get_positions(ph, tile_size, step)

    if sample_region:
        sx, sy, sw, sh = sample_region
        psx, psy = sx + overlap, sy + overlap
        x_positions = [
            x for x in x_positions
            if x < psx + sw and x + tile_size > psx
        ]
        y_positions = [
            y for y in y_positions
            if y < psy + sh and y + tile_size > sy
        ]
        out_x0, out_y0 = sx * scale, sy * scale
        out_w, out_h = sw * scale, sh * scale
    else:
        out_x0, out_y0 = 0, 0
        out_w, out_h = w * scale, h * scale

    output = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weights = np.zeros((out_h, out_w, 3), dtype=np.float64)

    total = len(x_positions) * len(y_positions)
    print(f"Processing {total} tiles ({len(x_positions)}x{len(y_positions)})")
    if sample_region:
        print(f"Sample region: input ({sx},{sy}) {sw}x{sh} -> output {out_w}x{out_h}")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipeline.set_progress_bar_config(disable=True)

    with tqdm(total=total, desc="Tiles", unit="tile") as pbar:
        for i, y in enumerate(y_positions):
            for j, x in enumerate(x_positions):
                tile_img = padded.crop((x, y, x + tile_size, y + tile_size))

                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=tile_img,
                    noise_level=noise_level,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]

                tile_arr = np.array(result, dtype=np.float64)

                # Smooth sliding-window blend mask (cosine ramps in overlap bands).
                mask_x = np.ones((out_tile,), dtype=np.float64)
                mask_y = np.ones((out_tile,), dtype=np.float64)
                out_step = step * scale
                ramp = max(0, out_tile - out_step)
                if ramp > 0:
                    ramp_vals = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp))
                    if x > x_positions[0]:
                        mask_x[:ramp] *= ramp_vals
                    if x < x_positions[-1]:
                        mask_x[-ramp:] *= ramp_vals[::-1]
                    if y > y_positions[0]:
                        mask_y[:ramp] *= ramp_vals
                    if y < y_positions[-1]:
                        mask_y[-ramp:] *= ramp_vals[::-1]
                mask_2d = mask_y[:, None] * mask_x[None, :]
                mask = np.repeat(mask_2d[:, :, None], 3, axis=2)

                ox = (x - overlap) * scale - out_x0
                oy = (y - overlap) * scale - out_y0

                src_x0 = max(0, -ox)
                src_y0 = max(0, -oy)
                dst_x0 = max(0, ox)
                dst_y0 = max(0, oy)
                src_x1 = min(out_tile, out_w - ox)
                src_y1 = min(out_tile, out_h - oy)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                dst_y1 = dst_y0 + (src_y1 - src_y0)

                output[dst_y0:dst_y1, dst_x0:dst_x1] += tile_arr[src_y0:src_y1, src_x0:src_x1] * mask[src_y0:src_y1, src_x0:src_x1]
                weights[dst_y0:dst_y1, dst_x0:dst_x1] += mask[src_y0:src_y1, src_x0:src_x1]
                pbar.update(1)

    weights = np.maximum(weights, 1e-8)
    output = (output / weights).clip(0, 255)

    if fidelity > 0.0:
        base_full = np.array(
            image.resize((w * scale, h * scale), resample=Image.Resampling.BICUBIC),
            dtype=np.float64,
        )
        if sample_region:
            base = base_full[out_y0:out_y0 + out_h, out_x0:out_x0 + out_w]
        else:
            base = base_full
        output = output * (1.0 - fidelity) + base * fidelity

    output = output.clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="low_res_cat.png")
    parser.add_argument("--output", default="tiled_upscale.png")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Tile stride in input pixels. Default uses tile_size-overlap.",
    )
    parser.add_argument("--noise-level", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--fidelity",
        type=float,
        default=0.0,
        help="Blend with bicubic base to preserve structure (0..1). Higher = fewer hallucinations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample", type=str, default=None,
        help="Sample region as x,y,w,h in input pixels (e.g. '100,200,200,200')"
    )
    parser.add_argument(
        "--prompt", default=(
            "Japanese movie poster, vibrant acrylic gouache illustration, "
            "woman in red sweater with colorful scarf in a supermarket, "
            "bold graphic design, clean sharp lines, crisp Japanese typography, "
            "high quality print, detailed illustration."
        ),
    )
    parser.add_argument(
        "--negative-prompt", default=(
            "blurry, artifacts, distortion, deformed text, "
            "watermark, low quality, noisy, smudged, canvas texture"
        ),
    )
    args = parser.parse_args()

    sample_region = None
    if args.sample:
        sample_region = tuple(int(v) for v in args.sample.split(","))

    print("Loading pipeline...")
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")
    pipeline.enable_attention_slicing()

    input_path = os.path.join("input_images", args.input)
    image = Image.open(input_path).convert("RGB")
    print(f"Input: {image.size[0]}x{image.size[1]}")

    result = tile_upscale(
        pipeline,
        image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        step=args.step,
        noise_level=args.noise_level,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
        sample_region=sample_region,
        fidelity=args.fidelity,
    )

    output_path = os.path.join("output_images", args.output)
    os.makedirs("output_images", exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path} ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    main()
