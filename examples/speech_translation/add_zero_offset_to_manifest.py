import argparse
import json
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main():
    args = get_args()
    with args.input.open() as in_f, args.output.open('w') as out_f:
        for i, line in enumerate(in_f):
            if i > 0:
                out_f.write('\n')
            data = json.loads(line)
            data["offset"] = 0.0
            out_f.write(json.dumps(data))


if __name__ == "__main__":
    main()
