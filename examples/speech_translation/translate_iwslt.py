import argparse
import re
from pathlib import Path

from nemo.collections.nlp.models import MTEncDecModel


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
SPACE_DEDUP = re.compile(r' +')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m")
    parser.add_argument("--model_path", "-p", type=Path)
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser()
    return args


def split_into_segments(text, max_num_chars_in_segment):
    segments = []
    start = 0
    prev_end_segment = 0
    prev_end_sent = 0
    for i in range(0, len(text)):
        if text[i] in [".", "?", "!"]:
            if i - start > max_num_chars_in_segment:
                segments.append(text[prev_end_segment:prev_end_sent])
                prev_end_segment = prev_end_sent
                prev_end_sent = i + 1
            else:
                prev_end_sent = i + 1
    if prev_end_sent < len(text):
        segments.append(text[prev_end_segment:])
    return segments


def main():
    args = get_args()
    pretrained = args.model_path if args.model is None else args.model
    model = MTEncDecModel.restore_from(pretrained)
    with open(args.input) as f:
        texts = f.readlines()
    processed = []
    max_num_chars_in_segment = 1024
    for text in texts:
        segments = split_into_segments(text, max_num_chars_in_segment)
        processed_segments = model.translate(segments, source_lang="en", target_lang="de")
        processed.append(SPACE_DEDUP.sub(' ', ' '.join(processed_segments)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for t in processed:
            f.write(t + '\n')


if __name__ == "__main__":
    main()
