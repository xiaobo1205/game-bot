"""Quick test: run template matching with both bobber templates against all screenshots."""

import time
import cv2
import numpy as np
from bot.vision import find_template


def test_match(image_path: str, template_path: str, thresholds: list[float]) -> None:
    frame = cv2.imread(image_path)
    template = cv2.imread(template_path)
    if frame is None or template is None:
        print(f"  ERROR: could not load {image_path} or {template_path}")
        return

    th, tw = template.shape[:2]
    fh, fw = frame.shape[:2]
    print(f"  Frame: {fw}x{fh}, Template: {tw}x{th}")

    for thresh in thresholds:
        t0 = time.perf_counter()
        matches = find_template(frame, template, thresh)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"    threshold={thresh:.2f}: {len(matches)} match(es), {elapsed_ms:.1f}ms", end="")
        if matches:
            for x, y in matches:
                print(f"  at ({x}, {y})", end="")
        print()


images = [
    "test_images/waiting.png",
    "test_images/splash1.png",
    "test_images/splash2.png",
]
templates = [
    "test_images/template1.png",
    "test_images/template2.png",
]
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for tpl in templates:
    print(f"\n{'='*60}")
    print(f"TEMPLATE: {tpl}")
    print(f"{'='*60}")
    for img in images:
        print(f"\n  vs {img}:")
        test_match(img, tpl, thresholds)

# Also save annotated images showing best matches
print(f"\n{'='*60}")
print("Saving annotated images...")
for tpl_path in templates:
    template = cv2.imread(tpl_path)
    tpl_name = tpl_path.split("/")[-1].replace(".png", "")
    for img_path in images:
        frame = cv2.imread(img_path)
        img_name = img_path.split("/")[-1].replace(".png", "")
        matches = find_template(frame, template, 0.6)
        annotated = frame.copy()
        th, tw = template.shape[:2]
        for i, (x, y) in enumerate(matches):
            x1, y1 = x - tw // 2, y - th // 2
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x1 + tw, y1 + th), color, 2)
            cv2.putText(annotated, f"#{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out = f"test_images/match_{tpl_name}_{img_name}.png"
        cv2.imwrite(out, annotated)
        print(f"  {out}: {len(matches)} match(es)")
