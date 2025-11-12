import base64
import io

from PIL import Image, ImageOps


def invert_image(image_base64: str):
    image_bytes = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(image_bytes)) as img:
        converted = ImageOps.invert(img.convert("RGB"))
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG")
        processed_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "processed_base64": processed_base64,
            "width": converted.width,
            "height": converted.height,
            "mode": converted.mode,
        }

