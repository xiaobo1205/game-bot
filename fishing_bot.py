"""CLI entry point for the WoW fishing bot (sound-triggered + visual bobber location)."""

import argparse
import sounddevice as sd

from bot.fishing import FishingBot


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
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    import os
    if not os.path.isdir(args.template_dir):
        parser.error(f"Template directory not found: {args.template_dir}")

    bot = FishingBot(
        template_dir=args.template_dir,
        threshold=args.threshold,
        volume_multiplier=args.volume_multiplier,
        cooldown=args.cooldown,
        loot_key=args.loot_key,
        monitor=args.monitor,
        audio_device=args.device,
        start_key=args.start_key,
        stop_key=args.stop_key,
        locate_delay=args.locate_delay,
    )
    bot.run()


if __name__ == "__main__":
    main()
