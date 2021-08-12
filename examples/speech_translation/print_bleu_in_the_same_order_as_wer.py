import json
import re
from argparse import ArgumentParser
from pathlib import Path

import yaml


ADD_COLON_RE = re.compile("(^[\\w\\.\\-]+|(?<=.\n|  )[\\w\\.\\-]+)(?=\n | [0-9])")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--bleu", "-b", type=Path, required=True)
    parser.add_argument("--wer_dir", "-d", type=Path, required=True)
    parser.add_argument("--keys", "-k", nargs=4, required=True)
    args = parser.parse_args()
    args.bleu = args.bleu.expanduser()
    args.wer_dir = args.wer_dir.expanduser()
    return args


def load_wer(dir_):
    result = {"not_segmented": {}, "segmented": {}}
    for k in result.keys():
        for e in (dir_ / Path(k)).iterdir():
            if str(e).endswith('.json'):
                with e.open() as f:
                    result[k][e.stem] = json.load(f)
    return result


def is_int(s):
    try:
        int(s)
    except ValueError:
        return False
    return True


def main():
    args = get_args()
    with args.bleu.open() as f:
        text = f.read()
    text = ADD_COLON_RE.sub(r'\1:', text)
    bleu = yaml.safe_load(text)
    wer = load_wer(args.wer_dir)
    order = list(wer[args.keys[0]][args.keys[3]].keys())
    for doc_id in order:
        print(doc_id)
    for doc_id in order:
        if is_int(doc_id):
            print(bleu[args.keys[0]][args.keys[1]][args.keys[2]][args.keys[3]][int(doc_id)])


if __name__ == "__main__":
    main()
