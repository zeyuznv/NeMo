import argparse
import re
from pathlib import Path

import nltk
from nemo.collections.nlp.models import MTEncDecModel


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
SPACE_DEDUP = re.compile(r' +')


nltk.download('punkt')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m")
    parser.add_argument("--model_path", "-p", type=Path)
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--one-sentence-segmentation", "-s", action="store_true")
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    if args.model_path is not None:
        args.model_path = args.model_path.expanduser()
    if args.model_path is None and args.model is None or args.model is not None and args.model_path is not None:
        raise ValueError(
            f"Exactly one of parameters `--model` and `--model-path` has to be provided. "
            f"`--model={args.model}`, `--model-path={args.model_path}`."
        )
    return args


def split_into_segments(text, max_num_chars_in_segment):
    segments = []
    prev_end_segment = 0
    prev_end_sent = 0
    for i in range(0, len(text)):
        if text[i] in [".", "?", "!"]:
            if i - prev_end_segment > max_num_chars_in_segment:
                segments.append(text[prev_end_segment:prev_end_sent])
                prev_end_segment = prev_end_sent
                prev_end_sent = i + 1
            else:
                prev_end_sent = i + 1
    if prev_end_segment < len(text):
        segments.append(text[prev_end_segment:])
    return segments


def main():
    args = get_args()
    if args.model is None:
        model = MTEncDecModel.restore_from(args.model_path)
    else:
        model = MTEncDecModel.from_pretrained(args.model)
    with open(args.input) as f:
        texts = f.readlines()
    processed = []
    max_num_chars_in_segment = 512
    for text in texts:
        segments = (
            nltk.sent_tokenize(text)
            if args.one_sentence_segmentation
            else split_into_segments(text, max_num_chars_in_segment)
        )
        processed_segments = model.translate(segments, source_lang="en", target_lang="de")
        processed.append(SPACE_DEDUP.sub(' ', ' '.join(processed_segments)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for t in processed:
            f.write(t + '\n')


if __name__ == "__main__":
    main()
