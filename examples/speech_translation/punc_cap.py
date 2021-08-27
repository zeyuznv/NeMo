import argparse
import json
import re
from pathlib import Path

from nemo.collections.nlp.models import PunctuationCapitalizationModel


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
SPACE_DEDUP = re.compile(r' +')
LONG_NUMBER = re.compile(r"[1-9][0-9]{3,}")
PUNCTUATION = re.compile("[.,?]")
DECIMAL = re.compile(f"[0-9]+{PUNCTUATION.pattern}? point({PUNCTUATION.pattern}? [0-9])+", flags=re.I)

MAX_NUM_SUBTOKENS_IN_INPUT = 8184


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-to-align-with", "-a", required=True, type=Path)
    parser.add_argument("--manifest-pred", "-p", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    args.manifest_to_align_with = args.manifest_to_align_with.expanduser()
    args.manifest_pred = args.manifest_pred.expanduser()
    args.output = args.output.expanduser()
    return args


def get_talk_id_order(manifest):
    talk_ids = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            m = TALK_ID_COMPILED_PATTERN.search(data["audio_filepath"])
            if m is None:
                raise ValueError(f"Talk id is not identified in file {manifest} for line {i}")
            talk_ids.append(m.group(0))
    return talk_ids


def load_manifest_text(manifest, text_key):
    result = {}
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            m = TALK_ID_COMPILED_PATTERN.search(data["audio_filepath"])
            if m is None:
                raise ValueError(f"Talk id is not identified in file {manifest} for line {i}")
            result[m.group(0)] = data[text_key]
    return result


def split_into_segments(text, size):
    segments = []
    for i in range(0, len(text), size):
        segments.append(text[i : i + size])
    return segments


def insert_commas_in_long_numbers(match):
    number = match.group(0)
    result = ""
    count = 0
    for i in range(0, len(number) - 3, 3):
        result = ',' + number[len(number) - i - 3 : len(number) - i] + result
        count += 3
    result = number[: len(number) - count] + result
    return result


def decimal_repl(match):
    text = PUNCTUATION.sub('', match.group(0))
    parts = text.split()
    return parts[0] + '.' + ''.join(parts[2:])


def main():
    args = get_args()
    model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
    order = get_talk_id_order(args.manifest_to_align_with)
    texts_to_process = load_manifest_text(args.manifest_pred, "pred_text")
    texts = [texts_to_process[talk_id] for talk_id in order]
    max_seq_len = 64
    processed = []
    processed_texts = model.add_punctuation_capitalization(
        texts, batch_size=MAX_NUM_SUBTOKENS_IN_INPUT // max_seq_len, max_seq_length=max_seq_len, step=8, margin=16
    )
    for text in processed_texts:
        processed.append(DECIMAL.sub(decimal_repl, SPACE_DEDUP.sub(' ', text)))
        # processed.append(
        #     LONG_NUMBER.sub(
        #         insert_commas_in_long_numbers,
        #         DECIMAL.sub(decimal_repl, SPACE_DEDUP.sub(' ', ' '.join(processed_segments))),
        #     )
        # )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for t in processed:
            f.write(t + '\n')


if __name__ == "__main__":
    main()
