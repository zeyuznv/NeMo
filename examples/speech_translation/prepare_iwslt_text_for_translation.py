import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup

from punc_cap import get_talk_id_order


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
SPACE_DEDUP = re.compile(r' +')
SOUNDS_DESCR = re.compile(r'^\([^)]+\)( \([^)]+\))*$')  # (Applause) (Laughter) (Silence):


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-to-align-with", "-a", required=True, type=Path)
    parser.add_argument("--text", "-t", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--join", "-j", action="store_true")
    args = parser.parse_args()
    args.manifest_to_align_with = args.manifest_to_align_with.expanduser()
    args.text = args.text.expanduser()
    args.output = args.output.expanduser()
    return args


def extract_segments(doc, join):
    if join:
        result = SPACE_DEDUP.sub(
            ' ', ' '.join([elem.text for elem in doc.findAll("seg") if not SOUNDS_DESCR.match(elem.text)])
        )
    else:
        result = [elem.text for elem in doc.findAll("seg") if not SOUNDS_DESCR.match(elem.text)]
    return result


def get_talk_id_to_text(src_text, join):
    with src_text.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    docs = soup.findAll("doc")
    result = {doc["docid"]: extract_segments(doc, join) for doc in docs}
    return result


def main():
    args = get_args()
    talk_id_to_text = get_talk_id_to_text(args.text, args.join)
    order = get_talk_id_order(args.manifest_to_align_with)
    with args.output.open('w') as f:
        for talk_id in order:
            if args.join:
                f.write(talk_id_to_text[talk_id] + '\n')
            else:
                for utterance in talk_id_to_text[talk_id]:
                    f.write(utterance + '\n')


if __name__ == "__main__":
    main()
