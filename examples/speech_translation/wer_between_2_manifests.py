import argparse
import json
from pathlib import Path

from nemo.collections.asr.metrics.wer import word_error_rate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=Path)
    parser.add_argument("ref", type=Path)
    args = parser.parse_args()
    args.hyp = args.hyp.expanduser()
    args.ref = args.ref.expanduser()
    return args


def read_texts_from_manifest(filepath, text_key):
    result = {}
    with filepath.open() as f:
        for line in f:
            data = json.loads(line)
            result[Path(data["audio_filepath"]).parts[-1]] = data[text_key]
    return result


def align_texts(hyps, refs, hyp_file, ref_file):
    aligned_h, aligned_r = [], []
    for k, v in hyps.items():
        aligned_h.append(v)
        if k not in refs:
            raise ValueError(
                f"Talk with id {k} is not found in references. Hypothesis file: {hyp_file}. Reference file: {ref_file}")
        aligned_r.append(refs[k])
    return aligned_h, aligned_r


def main():
    args = get_args()
    hyps = read_texts_from_manifest(args.hyp, "pred_text")
    refs = read_texts_from_manifest(args.ref, "text")
    if len(hyps) != len(refs):
        raise ValueError(f"Number of hypothesis texts {len(hyps)} in file {args.hyp} is not equal to number of "
                         f"reference texts {len(refs)} in file {args.ref}")
    hyps, refs = align_texts(hyps, refs, args.hyp, args.ref)
    print(word_error_rate(hyps, refs))


if __name__ == "__main__":
    main()
