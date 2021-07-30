from argparse import ArgumentParser
from pathlib import Path

from bs4 import BeautifulSoup


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    with args.input.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    result = ""
    for i, seg in enumerate(soup.findAll("seg")):
        result += seg.text.strip() + '\n'
    with args.output.open('w') as f:
        f.write(result)


if __name__ == "__main__":
    main()
