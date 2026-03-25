"""CLI entry point for the WoW fishing bot."""

import argparse
from bot.fishing import FishingBot


def main() -> None:
    parser = argparse.ArgumentParser(description="WoW Fishing Bot — template-matching bobber detector")
    parser.add_argument("--template", required=True, help="Path to bobber splash template image (PNG/JPG)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Match confidence 0-1 (default: 0.75)")
    parser.add_argument("--tick-rate", type=float, default=1.0, help="Seconds between scans (default: 1.0)")
    parser.add_argument("--loot-key", default="1", help="Key to press after catching (default: '1')")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (default: 1 = primary)")
    parser.add_argument("--start-key", default="f6", help="Hotkey to activate bot (default: F6)")
    parser.add_argument("--stop-key", default="f7", help="Hotkey to pause bot (default: F7)")
    args = parser.parse_args()

    bot = FishingBot(
        template_path=args.template,
        threshold=args.threshold,
        tick_rate=args.tick_rate,
        loot_key=args.loot_key,
        monitor=args.monitor,
        start_key=args.start_key,
        stop_key=args.stop_key,
    )
    bot.run()


if __name__ == "__main__":
    main()
