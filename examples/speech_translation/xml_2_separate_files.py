from argparse import ArgumentParser
from pathlib import Path

from bs4 import BeautifulSoup


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def main():
    args = get_args()
    with args.input.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    result = {}
    for doc in soup.findAll("doc"):
        doc_id = doc["docid"]
        result[doc_id] = ""
        for i, seg in enumerate(doc.findAll("seg")):
            result[doc_id] += seg.text.strip() + '\n'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in result.items():
        with (args.output_dir / Path(f"{doc_id}.txt")).open('w') as f:
            f.write(text)


if __name__ == "__main__":
    main()
