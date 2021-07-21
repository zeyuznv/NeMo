import argparse
import json
import re
from pathlib import Path

from nemo.collections.nlp.models import PunctuationCapitalizationModel


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")


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


def main():
    args = get_args()
    model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
    order = get_talk_id_order(args.manifest_to_align_with)
    texts_to_process = load_manifest_text(args.manifest_pred, "pred_text")
    texts = [texts_to_process[talk_id] for talk_id in order]
    processed = model.add_punctuation_capitalization(texts, batch_size=1, max_seq_length=10000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as f:
        for t in processed:
            f.write(t + '\n')


if __name__ == "__main__":
    main()
