from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    return parser.parse_args()

def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    file_paths = sorted([str(p.resolve()) for p in video_dir.rglob("*.mp4") if p.is_file()])
    with open(Path(args.output_dir) / "video_path.txt", "w") as f:
        for path in file_paths:
            f.write(path + "\n")

if __name__ == "__main__":
    main()