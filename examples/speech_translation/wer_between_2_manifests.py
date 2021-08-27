import argparse
import json
import re
from pathlib import Path

from nemo.collections.asr.metrics.wer import word_error_rate


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=Path)
    parser.add_argument("ref", type=Path)
    parser.add_argument("--output", "-o", type=Path, help="Path to output JSON file")
    args = parser.parse_args()
    args.hyp = args.hyp.expanduser()
    args.ref = args.ref.expanduser()
    if args.output is not None:
        args.output = args.output.expanduser()
    return args


def read_texts_from_manifest(filepath, text_key):
    result = {}
    with filepath.open() as f:
        for line in f:
            data = json.loads(line)
            result[Path(data["audio_filepath"]).parts[-1]] = data[text_key]
    return result


def align_texts(hyps, refs):
    aligned_h, aligned_r = [], []
    for k, v in hyps.items():
        aligned_h.append(v)
        aligned_r.append(refs[k])
    return aligned_h, aligned_r


def extract_talkid(filename):
    m = TALK_ID_COMPILED_PATTERN.search(str(filename))
    if m is None:
        raise ValueError(f"Cannot extract talkid from file name {filename}")
    else:
        result = m.group(0)
    return result


def get_wer_by_talkid(hyps, refs):
    result = {}
    for k, v in hyps.items():
        talkid = extract_talkid(k)
        result[talkid] = word_error_rate([v], [refs[k]])
    result = dict(sorted(result.items(), key=lambda x: -x[1]))
    return result


def check_input_data(hyps, refs, hyp_file, ref_file):
    if len(hyps) != len(refs):
        raise ValueError(
            f"Number of hypothesis texts {len(hyps)} in file {hyp_file} is not equal to number of "
            f"reference texts {len(refs)} in file {ref_file}"
        )
    if set(hyps.keys()) != set(refs.keys()):
        raise ValueError(
            f"File with hypothesis texts contains hypotheses for .wav files "
            f"{set(hyps.keys()) - set(refs.keys())} which are not present in reference file {ref_file}.\n"
            f"File with reference texts contains references for .wav files "
            f"{set(refs.keys()) - set(hyps.keys())} which are not present in hypothesis file {hyp_file}"
        )


def main():
    args = get_args()
    hyps = read_texts_from_manifest(args.hyp, "pred_text")
    refs = read_texts_from_manifest(args.ref, "text")
    check_input_data(hyps, refs, args.hyp, args.ref)
    aligned_hyps, aligned_refs = align_texts(hyps, refs)
    wer = word_error_rate(aligned_hyps, aligned_refs)
    print(wer)
    if args.output is not None:
        wer_by_talkid = get_wer_by_talkid(hyps, refs)
        wer_by_talkid["all"] = wer
        args.output.parent.mkdir(exist_ok=True, parents=True)
        with args.output.open('w') as f:
            json.dump(wer_by_talkid, f, indent=2)


if __name__ == "__main__":
    main()
