import argparse
import os
from PIL import Image, ImageOps, ImageDraw

def add_border(input_path, pct, dry_run=False):
    input_full_path = os.path.join("output_images", input_path)
    if not os.path.exists(input_full_path):
        input_full_path = os.path.join("input_images", input_path)
        if not os.path.exists(input_full_path):
            print(f"File not found in output_images or input_images: {input_path}")
            return

    try:
        img = Image.open(input_full_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    w, h = img.size
    
    # Calculate uniform border thickness based on the largest dimension
    # (pct / 100) gives the total percentage increase. 
    # We divide by 2 because the border is on both sides (left+right, top+bottom)
    # Example: 10% on a 1000px image = 100px total border -> 50px on each side
    border_px = int(max(w, h) * (pct / 100) / 2)
    
    # Add the uniform white border
    # ImageOps.expand adds pixels to all sides equally
    new_img = ImageOps.expand(img, border=border_px, fill='white')
    
    if dry_run:
        draw = ImageDraw.Draw(new_img)
        # Draw a black rectangle around the very edge
        width, height = new_img.size
        # outline width=5 for visibility
        draw.rectangle([(0,0), (width-1, height-1)], outline="black", width=5)
        print(f"Dry run: Added black outline. Border thickness is {border_px}px.")

    # Save
    base, ext = os.path.splitext(input_path)
    suffix = "_dry" if dry_run else ""
    # Remove directory from base for output filename if we want to save in current dir, 
    # but user didn't specify save location. For now, save alongside original.
    output_filename = f"{os.path.basename(base)}_bordered_{int(pct)}pct{suffix}{ext}"
    output_full_path = os.path.join("output_images", output_filename)
    os.makedirs("output_images", exist_ok=True)
    
    new_img.save(output_full_path)
    print(f"Saved to: {output_full_path}")
    print(f"Original: {w}x{h}")
    print(f"New: {new_img.size[0]}x{new_img.size[1]}")
    print(f"Uniform Border: {border_px}px on all sides")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a consistent white border to an image.")
    parser.add_argument("input_path", type=str, help="Path to input image")
    parser.add_argument("pct", type=float, help="Percentage size of the border relative to the image")
    parser.add_argument("--dry-run", action="store_true", help="Add a black outline for inspection")
    
    args = parser.parse_args()
    add_border(args.input_path, args.pct, args.dry_run)
