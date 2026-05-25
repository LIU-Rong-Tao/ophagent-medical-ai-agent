import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-name", default="overlay.png")
    parser.add_argument("--thumb-size", type=int, default=224)
    return parser.parse_args()


def read_image(path: Path, size: int):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.resize(img, (size, size))

    label = path.parent.name
    cv2.rectangle(img, (0, 0), (size, 26), (0, 0, 0), -1)
    cv2.putText(
        img,
        label,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def main():
    args = parse_args()

    input_root = Path(args.input_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_paths = sorted(input_root.glob(f"*/{args.image_name}"))

    if not overlay_paths:
        raise FileNotFoundError(f"No {args.image_name} found under {input_root}")

    imgs = [read_image(p, args.thumb_size) for p in overlay_paths]

    cols = 6
    rows = int(np.ceil(len(imgs) / cols))

    blank = np.zeros((args.thumb_size, args.thumb_size, 3), dtype=np.uint8)
    while len(imgs) < rows * cols:
        imgs.append(blank.copy())

    row_imgs = []
    for r in range(rows):
        row = imgs[r * cols : (r + 1) * cols]
        row_imgs.append(np.concatenate(row, axis=1))

    sheet = np.concatenate(row_imgs, axis=0)
    cv2.imwrite(str(output_path), sheet)

    print(f"Saved contact sheet: {output_path}")


if __name__ == "__main__":
    main()