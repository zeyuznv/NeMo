import re
from argparse import ArgumentParser
from pathlib import Path

from bs4 import BeautifulSoup


SOUNDS_DESCR = re.compile(r'^\([^)]+\)( \([^)]+\))*$')  # (Applause) (Laughter) (Silence):


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input-src", "-s", required=True, type=Path)
    parser.add_argument("--input-tgt", "-t", required=True, type=Path)
    parser.add_argument("--output-src", "-S", required=True, type=Path)
    parser.add_argument("--output-tgt", "-T", required=True, type=Path)
    args = parser.parse_args()
    args.input_src = args.input_src.expanduser()
    args.input_tgt = args.input_tgt.expanduser()
    args.output_src = args.output_src.expanduser()
    args.output_tgt = args.output_tgt.expanduser()
    return args


def read_and_filter(file_path):
    with file_path.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    docs = soup.findAll("doc")
    removed = {}
    for doc in docs:
        removed[doc["docid"]] = []
        count = 1
        for seg in doc.findAll("seg"):
            if SOUNDS_DESCR.match(seg.text):
                removed[doc["docid"]].append(seg["id"])
                seg.extract()
            else:
                seg["id"] = str(count)
                count += 1
    return soup, removed


def main():
    args = get_args()
    src, src_filtered_segments = read_and_filter(args.input_src)
    tgt, tgt_filtered_segments = read_and_filter(args.input_tgt)
    if src_filtered_segments != tgt_filtered_segments:
        docs_with_different_removals = {}
        for k in src_filtered_segments.keys():
            if src_filtered_segments[k] != tgt_filtered_segments[k]:
                docs_with_different_removals[k] = {"src": src_filtered_segments[k], "tgt": tgt_filtered_segments[k]}
        raise ValueError(
            f"Different segments were removed from source and target sets. Different removals: "
            f"{docs_with_different_removals}"
        )
    else:
        print("Removals:", src_filtered_segments)
    with args.output_src.open('w') as sf, args.output_tgt.open('w') as tf:
        sf.write(str(src))
        tf.write(str(tgt))


if __name__ == "__main__":
    main()
