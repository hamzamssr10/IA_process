"""Utility functions for drawing and image processing."""
import cv2


def draw_bbox_with_label(
    img,
    bbox,
    label,
    box_color=(0, 180, 255),
    text_color=(0, 0, 0),
    box_thickness=2,
    alpha=0.6
):
    """Draw bounding box with transparent rounded label."""
    x1, y1, x2, y2 = map(int, bbox)
    box_h = y2 - y1

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)

    # Dynamic font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(box_h / 120, 1.0))
    thickness = 1 if font_scale < 0.8 else 2

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad_x = int(th * 0.6)
    pad_y = int(th * 0.5)

    # Label position
    lx1 = x1
    ly2 = y1 - 4
    ly1 = ly2 - th - baseline - 2 * pad_y
    lx2 = x1 + tw + 2 * pad_x

    # If label goes outside top → move below box
    if ly1 < 0:
        ly1 = y1 + 4
        ly2 = ly1 + th + baseline + 2 * pad_y

    # Rounded transparent box
    radius = int((ly2 - ly1) * 0.4)
    radius = max(4, radius)

    overlay = img.copy()

    # Rectangles
    cv2.rectangle(overlay, (lx1 + radius, ly1), (lx2 - radius, ly2), box_color, -1)
    cv2.rectangle(overlay, (lx1, ly1 + radius), (lx2, ly2 - radius), box_color, -1)

    # Circles for rounded corners
    cv2.circle(overlay, (lx1 + radius, ly1 + radius), radius, box_color, -1)
    cv2.circle(overlay, (lx2 - radius, ly1 + radius), radius, box_color, -1)
    cv2.circle(overlay, (lx2 - radius, ly2 - radius), radius, box_color, -1)
    cv2.circle(overlay, (lx1 + radius, ly2 - radius), radius, box_color, -1)

    # Blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text
    text_x = lx1 + pad_x
    text_y = ly2 - pad_y - baseline

    cv2.putText(
        img,
        label,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return img


def box_center(x1, y1, x2, y2):
    """Calculate center of bounding box."""
    return int((x1 + x2) / 2), int((y1 + y2) / 2)
