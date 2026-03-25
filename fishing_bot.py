"""CLI entry point for the WoW fishing bot (sound-triggered + visual bobber location)."""

import argparse
import json
import os
from pathlib import Path

import cv2
import sounddevice as sd

from bot.fishing import FishingBot

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _load_screenshot(image_path: str | None) -> tuple[str | None, any]:
    """Load a screenshot from path or auto-detect from screenshots/ folder."""
    if not image_path:
        screenshots_dir = Path("screenshots")
        if screenshots_dir.is_dir():
            images = [f for f in screenshots_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
            if images:
                image_path = str(max(images, key=lambda f: f.stat().st_mtime))
                print(f"Auto-detected latest screenshot: {image_path}")

    if not image_path:
        print("Error: No screenshot found. Save a screenshot to screenshots/ first.")
        return None, None

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image: {image_path}")
        return None, None

    print(f"Loaded: {image_path}")
    return image_path, frame


def _prepare_display(frame):
    """Scale frame down for display if too large. Returns (display, scale)."""
    h, w = frame.shape[:2]
    max_display = 1600
    scale = min(max_display / w, max_display / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        display = frame.copy()
    return display, scale


def _select_rectangle(display, scale, title, instructions):
    """Interactive rectangle selection. Returns ROI dict or None."""
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

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse_callback)
    print(instructions)

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
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            print("Cancelled.")
            cv2.destroyWindow(title)
            return None
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
            cv2.destroyWindow(title)
            return {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}


def _select_point(display, scale, title, instructions):
    """Interactive point selection. Returns (x, y) in full-res coords or None."""
    state = {"clicked": False, "x": 0, "y": 0}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicked"] = True
            state["x"] = x
            state["y"] = y

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse_callback)
    print(instructions)

    while True:
        canvas = display.copy()
        if state["clicked"]:
            # Draw crosshair at selected point
            px, py = state["x"], state["y"]
            cv2.drawMarker(canvas, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            label = f"({int(px / scale)}, {int(py / scale)})"
            cv2.putText(canvas, label, (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            print("Cancelled.")
            cv2.destroyWindow(title)
            return None
        elif key == ord("r"):
            state["clicked"] = False
            print("Reset point.")
        elif key == 13:
            if not state["clicked"]:
                print("No point selected. Click on the fishing pole first.")
                continue
            cv2.destroyWindow(title)
            return (int(state["x"] / scale), int(state["y"] / scale))


def run_setup(image_path: str | None, config_path: str) -> None:
    """Two-step interactive setup: select bobber ROI, then fishing pole position.

    Both selections use the same screenshot. Results saved to config.json.
    """
    _, frame = _load_screenshot(image_path)
    if frame is None:
        return

    display, scale = _prepare_display(frame)

    # Step 1: Select bobber search area (ROI)
    print("\n" + "=" * 50)
    print("  STEP 1: Select BOBBER SEARCH AREA")
    print("=" * 50)
    roi = _select_rectangle(
        display, scale,
        "Step 1: Draw bobber search area - Enter to confirm, Esc to skip",
        "Draw a rectangle around the water area where the bobber appears.\n"
        "  - Click and drag to draw\n"
        "  - Enter to confirm, Esc to skip, R to reset"
    )

    # Step 2: Select fishing pole position
    print("\n" + "=" * 50)
    print("  STEP 2: Click on FISHING POLE in bags")
    print("=" * 50)
    pole_pos = _select_point(
        display, scale,
        "Step 2: Click on fishing pole - Enter to confirm, Esc to skip",
        "Click on the fishing pole in your inventory/bags.\n"
        "  - Left-click to select\n"
        "  - Enter to confirm, Esc to skip, R to reset"
    )

    cv2.destroyAllWindows()

    # Save to config
    path = Path(config_path)
    config = {}
    if path.exists():
        with open(path) as f:
            config = json.load(f)

    if roi:
        config["roi"] = roi
        print(f"ROI saved: {roi}")
    if pole_pos:
        config["pole_pos"] = {"x": pole_pos[0], "y": pole_pos[1]}
        print(f"Fishing pole position saved: {pole_pos}")

    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


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
    parser.add_argument("--setup", nargs="?", const="", default=None, metavar="IMAGE",
                        help="Interactive setup: select bobber ROI + fishing pole position. "
                             "Optional: path to screenshot (default: latest in screenshots/)")
    parser.add_argument("--bauble-interval", type=float, default=10.0,
                        help="Minutes between bauble applications (default: 10.0, 0 to disable)")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug screenshots showing match results")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.setup is not None:
        image = args.setup if args.setup else None
        run_setup(image, args.config)
        return

    if not os.path.isdir(args.template_dir):
        parser.error(f"Template directory not found: {args.template_dir}")

    # Load config file
    roi = None
    pole_pos = None
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        roi = config.get("roi")
        pole_pos = config.get("pole_pos")
        if roi:
            print(f"Loaded ROI: {roi}")
        if pole_pos:
            print(f"Loaded fishing pole position: ({pole_pos['x']}, {pole_pos['y']})")
    else:
        print(f"No config file ({args.config}). Run --setup to configure.")

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
        pole_pos=pole_pos,
        bauble_interval=args.bauble_interval * 60,  # convert min to sec
    )
    bot.run()


if __name__ == "__main__":
    main()
