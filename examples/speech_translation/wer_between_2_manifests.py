import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=Path)
    parser.add_argument("ref", type=Path)
    args = parser.parse_args()
    args.hyp = args.hyp.expanduser()
    args.ref = args.ref.expanduser()
    return args


def read_texts_from


def main():
    args = get_args()



if __name__ == "__main__":
    main()
