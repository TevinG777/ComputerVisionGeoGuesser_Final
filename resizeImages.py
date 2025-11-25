import argparse
import sys
from pathlib import Path

import cv2


def resize_images(
    input_dir: Path = Path("DataSet/Images"),
    output_dir: Path = Path("DataSet/Images_256"),
    size: tuple[int, int] = (256, 256),
) -> None:
    """
    Resize all images in ``input_dir`` to ``size`` using Lanczos interpolation and
    save them into ``output_dir``. Images are scaled (not cropped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    processed = 0
    skipped = 0

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in supported_exts:
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable file: {img_path.name}")
            skipped += 1
            continue

        resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        out_path = output_dir / img_path.name

        if not cv2.imwrite(str(out_path), resized):
            print(f"Failed to write resized image: {out_path}")
            skipped += 1
            continue

        processed += 1

    print(
        f"Finished. Resized {processed} image(s) to {size[0]}x{size[1]} "
        f"and skipped {skipped}. Output: {output_dir}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize all images in the dataset's Images folder to 256x256 using "
            "Lanczos scaling (no cropping)."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="DataSet/Images",
        type=Path,
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "--output-dir",
        default="DataSet/Images_256",
        type=Path,
        help="Directory to write the resized images to.",
    )
    parser.add_argument(
        "--size",
        default=[256, 256],
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Target size. Default: 256 256.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    resize_images(args.input_dir, args.output_dir, tuple(args.size))
