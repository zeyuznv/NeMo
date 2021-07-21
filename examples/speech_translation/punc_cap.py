import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", "-m", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    args.manifest = args.manifest.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    text = join_manifest_into_one_text


if __name__ == "__main__":
    main()
