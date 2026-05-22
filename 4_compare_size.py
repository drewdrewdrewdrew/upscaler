#!/usr/bin/env python3
"""Compare two images and report upscale factor."""
import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original", help="Original / low-res image")
    parser.add_argument("upscaled", help="Upscaled image")
    args = parser.parse_args()

    orig = Image.open(args.original)
    up = Image.open(args.upscaled)

    ow, oh = orig.size
    uw, uh = up.size
    orig_px = ow * oh
    up_px = uw * uh

    scale_x = uw / ow
    scale_y = uh / oh
    scale_linear = (scale_x + scale_y) / 2
    scale_area = (up_px / orig_px) ** 0.5

    print(f"Original:  {ow}×{oh} = {orig_px:,} px")
    print(f"Upscaled:  {uw}×{uh} = {up_px:,} px")
    print(f"Scale:     {scale_x:.2f}× (w) × {scale_y:.2f}× (h)")
    print(f"Linear:    ~{scale_linear:.1f}× upscaling")
    print(f"Area:      {up_px / orig_px:.1f}× pixels ({scale_area:.1f}× per axis)")


if __name__ == "__main__":
    main()
