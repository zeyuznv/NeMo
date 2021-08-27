import argparse
import json
import os
import re
from pathlib import Path


SPACE_DEDUP = re.compile(r' +')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-manifests-from-several-models",
        "-s",
        required=False,
        type=Path,
        help="Path to directory which contains directories with split manifests created by different models",
    )
    parser.add_argument("--split-manifests-from-one-model", "-S", type=Path)
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path to directory with joined manifests if `--split-manifests-from-several-models` is provided "
        "else if `--split-manifests-from-one-model` is provided this parameter is path to output manifest.",
    )
    parser.add_argument("--not-split-wav-dir", "-n", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.split_manifests_from_several_models is None
        and args.split_manifests_from_one_model is None
        or args.split_manifests_from_several_models is not None
        and args.split_manifests_from_one_model is not None
    ):
        raise ValueError(
            "Exactly one of parameters `--split-manifests-from-several-model` and `--split-manifests-from-one-model` "
            "has to be provided"
        )
    if args.split_manifests_from_several_models is not None:
        args.split_manifests_from_several_models = args.split_manifests_from_several_models.expanduser()
    if args.split_manifests_from_one_model is not None:
        args.split_manifests_from_one_model = args.split_manifests_from_one_model.expanduser()
    args.output = args.output.expanduser()
    args.not_split_wav_dir = args.not_split_wav_dir.expanduser()
    return args


def get_joined_text_and_duration(file_name):
    joined_text = ""
    joined_duration = None
    text_key = None
    with file_name.open() as f:
        data = sorted([json.loads(line) for line in f], key=lambda x: int(Path(x["audio_filepath"]).stem))
        if text_key is None:
            text_key = "text" if "text" in data[0] else "pred_text"
        for d in data:
            if joined_duration is None and "duration" in data:
                joined_duration = 0.0
            joined_text += ' ' + d[text_key]
            if joined_duration is not None:
                joined_duration += d["duration"]
    return SPACE_DEDUP.sub(' ', joined_text), joined_duration, text_key


def get_corresponding_not_split_wav_file(file_manifest, not_split_wav_dir):
    talk_id = file_manifest.parts[-1].split('.')[0]
    content = os.listdir(not_split_wav_dir)
    not_split_wav_file = None
    for x in content:
        if x.endswith(talk_id + '.wav'):
            not_split_wav_file = not_split_wav_dir / Path(x)
            break
    if not_split_wav_file is None:
        raise ValueError(
            f"No together file for manifest {file_manifest} was found in directory {not_split_wav_dir}. "
            f"Talk id: {talk_id}"
        )
    return not_split_wav_file


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def join_manifests(dir_, output_file, not_split_wav_dir):
    manifest = ""
    for elem in dir_.iterdir():
        if elem.is_file() and elem.suffix == ".manifest":
            joined_text, joined_duration, text_key = get_joined_text_and_duration(elem)
            not_split_wav_file = get_corresponding_not_split_wav_file(elem, not_split_wav_dir)
            if manifest:
                manifest += '\n'
            m_s = {"audio_filepath": str(not_split_wav_file), text_key: joined_text}
            if joined_duration is not None:
                m_s["duration"] = joined_duration
            manifest += json.dumps(m_s)
    with output_file.open('w') as f:
        f.write(manifest)


def main():
    args = get_args()
    if args.split_manifests_from_several_models is None:
        args.output.parent.mkdir(exist_ok=True, parents=True)
        join_manifests(args.split_manifests_from_one_model, args.output, args.not_split_wav_dir)
    else:
        args.output.mkdir(exist_ok=True, parents=True)
        for elem in args.split_manifests_from_several_models.iterdir():
            if elem.is_dir() and any([e.endswith('.manifest') for e in os.listdir(elem)]):
                join_manifests(elem, args.output / Path(elem.parts[-1] + '.manifest'), args.not_split_wav_dir)


if __name__ == "__main__":
    main()
