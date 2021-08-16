import argparse
import json
from pathlib import Path

from nemo.collections.nlp.models import PunctuationCapitalizationModel


def get_args():
    parser = argparse.ArgumentParser()
    input_ = parser.add_mutually_exclusive_group(required=True)
    input_.add_argument(
        "--input_manifest",
        "-m",
        type=Path,
        help="Path to the file with NeMo manifest which needs punctuation and capitalization. If the first element "
             "of manifest contains key 'pred_text', 'pred_text' values are passed for tokenization. Otherwise 'text' "
             "values are passed for punctuation and capitalization. Exactly one parameter of "
             "`--input_manifest` and `--input_text` should be provided."
    )
    input_.add_argument(
        "--input_text",
        "-t",
        type=Path,
        help="Path to file with text which needs punctuation and capitalization. Exactly one parameter of "
             "`--input_manifest` and `--input_text` should be provided."
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output_manifest",
        "-M",
        type=Path,
        help="Path to output NeMo manifest. Text with punctuation and capitalization will be under 'pred_text' key "
             "if 'pred_text' key is present in the first element of input manifest. Otherwise text with restored "
             "punctuation and capitalization will be under 'text' key. Exactly one parameter of "
             "`--output_manifest` and `--output_text` should be provided."
    )
    output.add_argument(
        "--output_text",
        "-T",
        type=Path,
        help="Path to file with text with restored punctuation and capitalization. Exactly one parameter of "
             "`--output_manifest` and `--output_text` should be provided."
    )
    model = parser.add_mutually_exclusive_group(required=False)
    model.add_argument("--pretrained_model", "-p")
    model.add_argument("--model_path", "-P", type=Path)
    parser.add_argument("--max_seq_length", "-L", type=int, default=64)
    parser.add_argument("--margin", "-g", type=int, default=16)
    parser.add_argument("--step", "-s", type=int, default=8)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    args = parser.parse_args()
    if args.input_manifest is None and args.output_manifest is not None:
        parser.error("--output_manifest requires --input_manifest")
    if args.pretrained_name is None and args.model_path is None:
        args.pretrained_name = "punctuation_en_bert"
    for name in ["input_manifest", "input_text", "output_manifest", "output_text", "model_path"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser())
    return args


def load_manifest(manifest):
    result = []
    with manifest.open() as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            result.append(data)
    return result


def main():
    args = get_args()
    if args.pretrained_name is None:
        model = PunctuationCapitalizationModel.restore_from(args.model_path)
    else:
        model = PunctuationCapitalizationModel.from_pretrained(args.pretrained_name)
    if args.input_manifest is None:
        texts = []
        with args.input_text.open() as f:
            texts.append(f.readline().strip())
    else:
        manifest = load_manifest(args.input_manifest)
        text_key = "pred_text" if "pred_text" in manifest[0] else "text"
        texts = []
        for item in manifest:
            texts.append(item[text_key])
    processed_texts = model.add_punctuation_capitalization(
        texts, batch_size=args.batch_size, max_seq_length=args.max_seq_length, step=8, margin=16,
    )
    if args.output_manifest is None:
        with args.output_text.open('w') as f:
            for t in processed_texts:
                f.write(t + '\n')
    else:
        with args.output_manifest.open('w') as f:
            for item, t in zip(manifest, processed_texts):
                item[text_key] = t
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()
