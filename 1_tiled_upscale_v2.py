import argparse
import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline


def tile_upscale_latent(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    tile_size: int = 128,
    overlap: int = 32,
    noise_level: int = 10,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 50,
    seed: int = 42,
    sample_region: tuple = None,
):
    """
    Upscale 4x via tiled diffusion, blending in latent space
    and decoding the full stitched latent in one VAE pass.
    """
    w, h = image.size
    scale = 4
    # In this pipeline, latent spatial dims == input spatial dims (1:1)
    # VAE decode then does the 4x spatial upscale
    latent_tile = tile_size
    latent_overlap = overlap

    step = tile_size - overlap

    def get_positions(length, tile, step):
        positions = list(range(0, length - tile, step))
        if not positions or positions[-1] + tile < length:
            positions.append(max(0, length - tile))
        return positions

    x_positions = get_positions(w, tile_size, step)
    y_positions = get_positions(h, tile_size, step)

    if sample_region:
        sx, sy, sw, sh = sample_region
        x_positions = [
            x for x in x_positions
            if x < sx + sw and x + tile_size > sx
        ]
        y_positions = [
            y for y in y_positions
            if y < sy + sh and y + tile_size > sy
        ]
        lat_x0, lat_y0 = sx, sy
        lat_w, lat_h = sw, sh
    else:
        lat_x0, lat_y0 = 0, 0
        lat_w, lat_h = w, h

    num_channels = pipeline.vae.config.latent_channels  # 4
    full_latents = torch.zeros(1, num_channels, lat_h, lat_w, dtype=torch.float16, device="cuda")
    full_weights = torch.zeros(1, 1, lat_h, lat_w, dtype=torch.float16, device="cuda")

    total = len(x_positions) * len(y_positions)
    print(f"Processing {total} tiles ({len(x_positions)}x{len(y_positions)})")
    if sample_region:
        print(f"Sample region: input ({sx},{sy}) {sw}x{sh} -> latent {lat_w}x{lat_h} -> output {lat_w*scale}x{lat_h*scale}")

    # Pre-generate full noise tensor for consistency across tiles
    generator = torch.Generator(device="cuda").manual_seed(seed)
    full_noise = torch.randn(1, num_channels, h, w, generator=generator, device="cuda", dtype=torch.float16)

    for i, y in enumerate(y_positions):
        for j, x in enumerate(x_positions):
            idx = i * len(x_positions) + j + 1
            print(f"  Tile {idx}/{total} at input ({x},{y})")

            tile_img = image.crop((x, y, x + tile_size, y + tile_size))

            # Slice noise for this tile from the shared tensor
            tile_noise = full_noise[:, :, y:y + latent_tile, x:x + latent_tile].clone()

            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=tile_img,
                noise_level=noise_level,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                latents=tile_noise,
                output_type="latent",
            )
            tile_latents = result.images  # (1, 4, latent_tile, latent_tile)

            # Build feathered blend mask in latent space
            mask = torch.ones(1, 1, latent_tile, latent_tile, dtype=torch.float16, device="cuda")
            ramp = latent_overlap

            if ramp > 0:
                ramp_vals = torch.linspace(0, 1, ramp, device="cuda", dtype=torch.float16)
                if x > x_positions[0]:
                    mask[:, :, :, :ramp] *= ramp_vals[None, None, None, :]
                if x < x_positions[-1]:
                    mask[:, :, :, -ramp:] *= ramp_vals.flip(0)[None, None, None, :]
                if y > y_positions[0]:
                    mask[:, :, :ramp, :] *= ramp_vals[None, None, :, None]
                if y < y_positions[-1]:
                    mask[:, :, -ramp:, :] *= ramp_vals.flip(0)[None, None, :, None]

            # Place into full latent, offset by sample region
            dst_lx = x - lat_x0
            dst_ly = y - lat_y0

            # Clip to bounds
            s_x0 = max(0, -dst_lx)
            s_y0 = max(0, -dst_ly)
            d_x0 = max(0, dst_lx)
            d_y0 = max(0, dst_ly)
            s_x1 = min(latent_tile, lat_w - dst_lx)
            s_y1 = min(latent_tile, lat_h - dst_ly)
            d_x1 = d_x0 + (s_x1 - s_x0)
            d_y1 = d_y0 + (s_y1 - s_y0)

            full_latents[:, :, d_y0:d_y1, d_x0:d_x1] += tile_latents[:, :, s_y0:s_y1, s_x0:s_x1] * mask[:, :, s_y0:s_y1, s_x0:s_x1]
            full_weights[:, :, d_y0:d_y1, d_x0:d_x1] += mask[:, :, s_y0:s_y1, s_x0:s_x1]

    full_weights = torch.clamp(full_weights, min=1e-8)
    full_latents = full_latents / full_weights

    # Single VAE decode pass
    print("Decoding stitched latents...")
    pipeline.enable_vae_tiling()
    with torch.no_grad():
        decoded = pipeline.vae.decode(full_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]

    # Post-process: clamp to [0,1], convert to uint8
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    decoded = (decoded * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="low_res_cat.png")
    parser.add_argument("--output", default="tiled_upscale_v2.png")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--noise-level", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
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

    result = tile_upscale_latent(
        pipeline,
        image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        noise_level=args.noise_level,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
        sample_region=sample_region,
    )

    output_path = os.path.join("output_images", args.output)
    os.makedirs("output_images", exist_ok=True)
    result.save(output_path)
    print(f"Saved: {output_path} ({result.size[0]}x{result.size[1]})")


if __name__ == "__main__":
    main()
