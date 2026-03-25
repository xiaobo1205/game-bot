"""Test color detection against fishing screenshots.

Usage:
    python test_detection.py --waiting img1.png --splash img2.png img3.png --template template.png

The template should be a cropped region of the bobber splash (from one of the splash images).
If --template is not provided, the script will auto-crop a region from the first splash image
around the bobber area for testing.
"""

import argparse
import time

import cv2
import numpy as np

from bot.vision import extract_hsv_range, find_color_regions, find_template


def analyze_image(name: str, frame: np.ndarray, lower_hsv, upper_hsv, min_area: int = 200):
    """Run color detection on a single image and report results."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"  Image size: {frame.shape[1]}x{frame.shape[0]}")

    # Time the color detection
    t0 = time.perf_counter()
    regions = find_color_regions(frame, lower_hsv, upper_hsv, min_area=min_area)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  Detection time: {elapsed_ms:.2f}ms")
    print(f"  Regions found: {len(regions)}")

    for i, r in enumerate(sorted(regions, key=lambda r: r["area"], reverse=True)):
        print(f"    Region #{i+1}: center=({r['x']}, {r['y']}), "
              f"size={r['w']}x{r['h']}, area={r['area']}")

    return regions, elapsed_ms


def analyze_with_template_matching(name: str, frame: np.ndarray, template: np.ndarray, threshold: float = 0.75):
    """Run template matching for comparison."""
    t0 = time.perf_counter()
    matches = find_template(frame, template, threshold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  [Template matching comparison] threshold={threshold}")
    print(f"  Detection time: {elapsed_ms:.2f}ms")
    print(f"  Matches found: {len(matches)}")
    for i, (x, y) in enumerate(matches):
        print(f"    Match #{i+1}: ({x}, {y})")

    return matches, elapsed_ms


def save_annotated(path: str, frame: np.ndarray, regions: list, label: str):
    """Save image with green rectangles around detected regions."""
    annotated = frame.copy()
    for i, r in enumerate(sorted(regions, key=lambda r: r["area"], reverse=True)):
        x1 = r["x"] - r["w"] // 2
        y1 = r["y"] - r["h"] // 2
        x2 = x1 + r["w"]
        y2 = y1 + r["h"]
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"#{i+1} area={r['area']}"
        cv2.putText(annotated, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(path, annotated)
    print(f"  Saved annotated: {path}")


def main():
    parser = argparse.ArgumentParser(description="Test color detection on fishing screenshots")
    parser.add_argument("--waiting", required=True, help="Screenshot of bobber waiting (no splash)")
    parser.add_argument("--splash", nargs="+", required=True, help="Screenshot(s) of bobber splash")
    parser.add_argument("--template", help="Cropped template of bobber splash. If not provided, uses splash center crop.")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum pixel area (default: 100)")
    parser.add_argument("--save-annotated", action="store_true", help="Save annotated images to test_images/")
    args = parser.parse_args()

    # Load images
    waiting = cv2.imread(args.waiting)
    splashes = [cv2.imread(p) for p in args.splash]

    if waiting is None:
        print(f"Error: Could not load waiting image: {args.waiting}")
        return
    for i, s in enumerate(splashes):
        if s is None:
            print(f"Error: Could not load splash image: {args.splash[i]}")
            return

    # Load or auto-crop template
    if args.template:
        template = cv2.imread(args.template)
        if template is None:
            print(f"Error: Could not load template: {args.template}")
            return
        print(f"Template loaded from: {args.template}")
    else:
        # Auto-crop: take a 100x80 region around approximate bobber location from first splash
        # The bobber is typically in the upper-center area of the screen
        h, w = splashes[0].shape[:2]
        cx, cy = w // 2 - 50, h // 4  # rough bobber area
        crop_w, crop_h = 120, 100
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        template = splashes[0][y1:y1+crop_h, x1:x1+crop_w]
        print(f"Auto-cropped template from splash[0] at ({x1},{y1}) size {crop_w}x{crop_h}")
        cv2.imwrite("test_images/auto_template.png", template)

    # Extract HSV range from template
    print(f"\nTemplate size: {template.shape[1]}x{template.shape[0]}")
    lower_hsv, upper_hsv = extract_hsv_range(template)
    print(f"Extracted HSV range:")
    print(f"  Lower: H={lower_hsv[0]}, S={lower_hsv[1]}, V={lower_hsv[2]}")
    print(f"  Upper: H={upper_hsv[0]}, S={upper_hsv[1]}, V={upper_hsv[2]}")

    # Show template HSV coverage
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    template_mask = cv2.inRange(template_hsv, np.array(lower_hsv), np.array(upper_hsv))
    match_pct = np.count_nonzero(template_mask) / template_mask.size * 100
    print(f"  Template self-coverage: {match_pct:.1f}%")

    # Test on all images
    print("\n" + "="*60)
    print("TESTING COLOR DETECTION")
    print("="*60)

    # Test different min_area values
    for min_area in [50, 100, 200, 500]:
        print(f"\n--- min_area = {min_area} ---")

        regions_w, time_w = analyze_image(f"WAITING - {args.waiting}", waiting, lower_hsv, upper_hsv, min_area)
        for i, sp in enumerate(splashes):
            regions_s, time_s = analyze_image(f"SPLASH #{i+1} - {args.splash[i]}", sp, lower_hsv, upper_hsv, min_area)

        # Report false positives
        if regions_w:
            print(f"\n  WARNING: {len(regions_w)} false positive(s) on waiting image!")
        else:
            print(f"\n  OK: No false positives on waiting image")

    # Also run template matching for comparison
    print("\n" + "="*60)
    print("COMPARISON: TEMPLATE MATCHING")
    print("="*60)

    for threshold in [0.6, 0.7, 0.8]:
        print(f"\n--- threshold = {threshold} ---")
        analyze_with_template_matching(f"WAITING", waiting, template, threshold)
        for i, sp in enumerate(splashes):
            analyze_with_template_matching(f"SPLASH #{i+1}", sp, template, threshold)

    # Save annotated images if requested
    if args.save_annotated:
        print("\n\nSaving annotated images...")
        regions_w, _ = analyze_image("waiting", waiting, lower_hsv, upper_hsv, args.min_area)
        save_annotated("test_images/annotated_waiting.png", waiting, regions_w, "WAITING")
        for i, sp in enumerate(splashes):
            regions_s, _ = analyze_image(f"splash_{i+1}", sp, lower_hsv, upper_hsv, args.min_area)
            save_annotated(f"test_images/annotated_splash_{i+1}.png", sp, regions_s, f"SPLASH #{i+1}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Color-based detection relies on the template's HSV signature.")
    print("Key tuning parameters:")
    print("  --min-area: Raise to reduce false positives, lower for better sensitivity")
    print("  std_devs in extract_hsv_range(): Wider range catches more but risks false positives")


if __name__ == "__main__":
    main()
