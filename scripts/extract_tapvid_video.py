import argparse
from heft.data.tapvid import TapVid, TapVidRGBStack, TapVidKinetics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["davis", "kinetics", "rgb_stack"])
    parser.add_argument("--output-dir", type=str, default="datasets")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.dataset == "davis":
        tapvid = TapVid(root=args.dataset_dir)
    elif args.dataset == "kinetics":
        tapvid = TapVidKinetics(root=args.dataset_dir)
    elif args.dataset == "rgb_stack":
        tapvid = TapVidRGBStack(root=args.dataset_dir)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    tapvid.save_videos(output_dir=args.output_dir)
    print(f"Saved videos to {args.output_dir}")

if __name__ == "__main__":
    main()