from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent

# Crop boxes are in PIL format: (left, upper, right, lower).
# Adjust these values if the paper layout needs a tighter crop.
CROP_BOXES = {
    "fig_exp_setup1.png": (135, 0, 2679, 778),
    "fig_exp_setup2.png": (0, 50, 2544, 828),
}


def crop_image(filename: str, crop_box: tuple[int, int, int, int]) -> None:
    src = HERE / filename
    dst_png = src.with_name(f"{src.stem}_cropped.png")
    dst_pdf = src.with_name(f"{src.stem}_cropped.pdf")

    with Image.open(src) as image:
        print(f"{src.name}: original size = {image.size}")
        print(f"{src.name}: crop box = {crop_box}")

        cropped = image.crop(crop_box)
        print(f"{dst_png.name}: cropped size = {cropped.size}")

        cropped.save(dst_png)
        cropped.convert("RGB").save(dst_pdf, "PDF", resolution=300.0)
        print(f"{dst_pdf.name}: saved")


def main() -> None:
    for filename, crop_box in CROP_BOXES.items():
        crop_image(filename, crop_box)


if __name__ == "__main__":
    main()
