"""CLI entry point for the WoW fishing bot (sound-triggered + visual bobber location)."""

import argparse
import json
import os
from pathlib import Path

import cv2
import sounddevice as sd

from bot.fishing import FishingBot

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def select_roi(image_path: str | None, config_path: str) -> dict | None:
    """Open an interactive window to draw an ROI rectangle on a screenshot.

    Args:
        image_path: Path to screenshot file, or None to auto-detect from screenshots/.
        config_path: Path to save the ROI config JSON.

    Returns:
        ROI dict or None if cancelled.
    """
    # Auto-detect latest screenshot if no path given
    if not image_path:
        screenshots_dir = Path("screenshots")
        if screenshots_dir.is_dir():
            images = [f for f in screenshots_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
            if images:
                image_path = str(max(images, key=lambda f: f.stat().st_mtime))
                print(f"Auto-detected latest screenshot: {image_path}")

    if not image_path:
        print("Error: No screenshot found. Save a screenshot to screenshots/ first.")
        return None

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    print(f"Loaded: {image_path}")

    # Scale down for display if too large
    h, w = frame.shape[:2]
    max_display = 1600
    scale = min(max_display / w, max_display / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        display = frame.copy()

    state = {"drawing": False, "start": (0, 0), "end": (0, 0)}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["end"] = (x, y)

    window_name = "Draw ROI - Enter to save, Esc to cancel, R to reset"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Draw a rectangle around the area where the bobber appears.")
    print("  - Click and drag to draw")
    print("  - Press Enter to save")
    print("  - Press Escape to cancel")
    print("  - Press R to reset")

    roi = None
    while True:
        canvas = display.copy()

        if state["start"] != state["end"]:
            cv2.rectangle(canvas, state["start"], state["end"], (0, 255, 0), 2)
            rx = abs(state["end"][0] - state["start"][0])
            ry = abs(state["end"][1] - state["start"][1])
            label = f"{int(rx / scale)}x{int(ry / scale)}px"
            cv2.putText(canvas, label,
                        (min(state["start"][0], state["end"][0]),
                         min(state["start"][1], state["end"][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            print("Cancelled.")
            break
        elif key == ord("r"):
            state["start"] = (0, 0)
            state["end"] = (0, 0)
            print("Reset rectangle.")
        elif key == 13:
            if state["start"] == state["end"]:
                print("No rectangle drawn. Draw one first.")
                continue

            x1 = int(min(state["start"][0], state["end"][0]) / scale)
            y1 = int(min(state["start"][1], state["end"][1]) / scale)
            x2 = int(max(state["start"][0], state["end"][0]) / scale)
            y2 = int(max(state["start"][1], state["end"][1]) / scale)

            roi = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}

            path = Path(config_path)
            config = {}
            if path.exists():
                with open(path) as f:
                    config = json.load(f)
            config["roi"] = roi
            with open(path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"ROI saved to {config_path}: {roi}")
            break

    cv2.destroyAllWindows()
    return roi


def list_devices() -> None:
    """Print all available audio devices."""
    print("Available audio devices:")
    print("-" * 70)
    for i, dev in enumerate(sd.query_devices()):
        direction = ""
        if dev["max_input_channels"] > 0:
            direction += "IN"
        if dev["max_output_channels"] > 0:
            direction += "/OUT" if direction else "OUT"
        print(f"  [{i:2d}] {dev['name']:<50s} {direction}")
    print("-" * 70)

    # Show host APIs
    print("\nHost APIs:")
    for i, api in enumerate(sd.query_hostapis()):
        default = " (default)" if i == sd.default.hostapi else ""
        print(f"  [{i}] {api['name']}{default}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WoW Fishing Bot — sound-triggered with visual bobber detection"
    )
    parser.add_argument("--template-dir", default="templates",
                        help="Directory containing bobber template images (default: templates/)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Template match confidence 0-1 (default: 0.7)")
    parser.add_argument("--volume-multiplier", type=float, default=3.0,
                        help="Volume spike threshold as multiplier of ambient (default: 3.0)")
    parser.add_argument("--cooldown", type=float, default=3.0,
                        help="Seconds between catch attempts (default: 3.0)")
    parser.add_argument("--cast-delay", type=float, default=2.0,
                        help="Seconds to wait after casting for bobber to land (default: 2.0)")
    parser.add_argument("--loot-key", default="1",
                        help="Key to press after catching (default: '1')")
    parser.add_argument("--monitor", type=int, default=1,
                        help="Monitor index (default: 1 = primary)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio device index (default: auto-detect loopback)")
    parser.add_argument("--start-key", default="f6",
                        help="Hotkey to activate bot (default: F6)")
    parser.add_argument("--stop-key", default="f7",
                        help="Hotkey to pause bot (default: F7)")
    parser.add_argument("--locate-delay", type=float, default=1.0,
                        help="Seconds to wait before locating bobber after F6 (default: 1.0)")
    parser.add_argument("--config", default="config.json",
                        help="Config file with ROI settings (default: config.json)")
    parser.add_argument("--select-roi", nargs="?", const="", default=None, metavar="IMAGE",
                        help="Open ROI selector. Optional: path to screenshot (default: latest in screenshots/)")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug screenshots showing match results")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.select_roi is not None:
        image = args.select_roi if args.select_roi else None
        select_roi(image, args.config)
        return

    if not os.path.isdir(args.template_dir):
        parser.error(f"Template directory not found: {args.template_dir}")

    # Load ROI from config file if it exists
    roi = None
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        roi = config.get("roi")
        if roi:
            print(f"Loaded ROI from {args.config}: {roi}")
    else:
        print(f"No config file ({args.config}). Run --select-roi to set ROI.")

    bot = FishingBot(
        template_dir=args.template_dir,
        threshold=args.threshold,
        volume_multiplier=args.volume_multiplier,
        cooldown=args.cooldown,
        loot_key=args.loot_key,
        cast_delay=args.cast_delay,
        monitor=args.monitor,
        audio_device=args.device,
        start_key=args.start_key,
        stop_key=args.stop_key,
        locate_delay=args.locate_delay,
        debug=args.debug,
        roi=roi,
    )
    bot.run()


if __name__ == "__main__":
    main()
