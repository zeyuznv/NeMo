import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup

from punc_cap import get_talk_id_order


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
NOT_TRANSCRIPT_PATTERN = re.compile(r"[^a-z ']")
SPACE_DEDUP = re.compile(r' +')
SOUNDS_DESCR = re.compile(r'^\([^)]+\)$')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-to-align-with", "-a", required=True, type=Path)
    parser.add_argument("--text", "-t", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    args.manifest_to_align_with = args.manifest_to_align_with.expanduser()
    args.text = args.text.expanduser()
    args.output = args.output.expanduser()
    return args


def get_talk_id_to_text(src_text):
    with src_text.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    docs = soup.findAll("doc")
    result = {
        doc["docid"]:
            SPACE_DEDUP.sub(
                ' ', ' '.join(
                    [elem.text for elem in doc.findAll("seg")
                     if not SOUNDS_DESCR.match(elem.text)])
            )
        for doc in docs
    }
    return result


def main():
    args = get_args()
    talk_id_to_text = get_talk_id_to_text(args.text)
    order = get_talk_id_order(args.manifest_to_align_with)
    with args.output.open('w') as f:
        for talk_id in order:
            f.write(talk_id_to_text[talk_id] + '\n')


if __name__ == "__main__":
    main()
