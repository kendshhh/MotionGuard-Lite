from __future__ import annotations

import sys
from pathlib import Path

import cv2 as cv


WINDOW_NAME = "Technique Reference"
MAX_WIDTH = 980
MAX_HEIGHT = 720


def resize_to_fit(image):
    height, width = image.shape[:2]
    if width <= MAX_WIDTH and height <= MAX_HEIGHT:
        return image

    scale = min(MAX_WIDTH / max(width, 1), MAX_HEIGHT / max(height, 1))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    return cv.resize(image, (resized_width, resized_height), interpolation=cv.INTER_AREA)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: reference_pose_viewer.py <image_path>")
        return 1

    image_path = Path(sys.argv[1]).resolve()
    if not image_path.is_file():
        print(f"Reference image was not found at '{image_path}'.")
        return 1

    image = cv.imread(str(image_path))
    if image is None:
        print(f"Reference image could not be loaded from '{image_path}'. Use a PNG, JPG, JPEG, WEBP, or BMP file.")
        return 1

    image = resize_to_fit(image)

    try:
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        cv.imshow(WINDOW_NAME, image)
        cv.waitKey(0)
    except cv.error as exc:
        print(f"OpenCV could not show the reference pose window: {exc}")
        return 1
    finally:
        cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())