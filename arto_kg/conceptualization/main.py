"""
Command-line entry point for artwork conceptualization.
"""

import argparse
import traceback

from .main_pipeline import create_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="ARTO conceptualization pipeline")
    parser.add_argument("--style", help="Fixed art style to generate")
    parser.add_argument("--count", "--num-artworks", "--num_artworks", dest="count", type=int, default=1,
                        help="Number of artworks to generate")
    parser.add_argument(
        "--num-objects",
        "--num_objects",
        "--max-secondary-objects",
        "--max_secondary_objects",
        dest="max_secondary_objects",
        type=int,
        default=5,
        help="Maximum number of secondary objects to sample",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="data/output",
                        help="Output directory")
    parser.add_argument("--detailed-output", action="store_true",
                        help="Save single-artwork runs with stage folders and final_results output")
    parser.add_argument("--disable-batch-mode", action="store_true",
                        help="Force sequential generation when count > 1")
    parser.add_argument("--no-save-intermediate", action="store_true",
                        help="Disable intermediate batch result files")
    return parser


def main() -> None:
    """Run conceptualization CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    batch_mode = args.count > 1 and not args.disable_batch_mode
    pipeline = create_pipeline(batch_mode=batch_mode, output_dir=args.output_dir)

    try:
        if args.count == 1:
            result = pipeline.generate_single_artwork(
                style=args.style,
                max_secondary_objects=args.max_secondary_objects,
                output_dir=args.output_dir,
                use_detailed_output=args.detailed_output,
            )
            if "error" in result:
                print(f"[ERROR] Conceptualization failed: {result['error']}")
                raise SystemExit(1)

            print(f"[INFO] Generated artwork: {result.get('artwork_id', 'unknown')}")
            print(f"[INFO] Output directory: {args.output_dir}")
            raise SystemExit(0)

        results = pipeline.generate_batch_artworks(
            count=args.count,
            styles=[args.style] if args.style else None,
            max_secondary_objects=args.max_secondary_objects,
            output_dir=args.output_dir,
            save_intermediate=not args.no_save_intermediate,
        )

        success_count = len([item for item in results if "error" not in item])
        failed_count = len(results) - success_count

        print(f"[INFO] Generated {success_count}/{len(results)} artworks")
        print(f"[INFO] Output directory: {args.output_dir}")

        if failed_count == 0:
            raise SystemExit(0)
        if success_count > 0:
            raise SystemExit(1)
        raise SystemExit(2)

    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL] Conceptualization failed: {exc}")
        traceback.print_exc()
        raise SystemExit(3)
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
