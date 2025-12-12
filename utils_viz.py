from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image: Image.Image,
    boxes: List[Tuple[float, float, float, float]],
    labels: List[int],
    scores: List[float],
    score_thr: float = 0.5,
    class_names: List[str] = None,
) -> Image.Image:
    """Draw detection boxes on a copy of the image and return it.

    boxes: list of (xmin, ymin, xmax, ymax)
    labels: list of int class ids
    scores: list of float confidences
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), cls_id, score in zip(boxes, labels, scores):
        if score < score_thr:
            continue
        color = (255, 0, 0)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        label_text = f"{cls_id}"
        if class_names and 0 <= cls_id < len(class_names):
            label_text = class_names[cls_id]
        label = f"{label_text}:{score:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([(x1, y1 - th - 2), (x1 + tw + 2, y1)], fill=color)
        draw.text((x1 + 1, y1 - th - 1), label, fill=(255, 255, 255), font=font)
    return img
