import argparse
import json
import os
import re
from pathlib import Path


SPACE_DEDUP = re.compile(r' +')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifests", "-s", required=True, type=Path)
    parser.add_argument("--output-dir", "-o", required=True, type=Path)
    parser.add_argument("--not-split-wav-dir", "-n", required=True, type=Path)
    args = parser.parse_args()
    args.split_manifests = args.split_manifests.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.not_split_wav_dir = args.not_split_wav_dir.expanduser()
    return args


def get_joined_text_and_duration(file_name):
    joined_text = ""
    joined_duration = 0.0
    with file_name.open() as f:
        for line in f:
            data = json.loads(line)
            joined_text += ' ' + data["text"]
            joined_duration += data["duration"]
    return SPACE_DEDUP.sub(' ', joined_text), joined_duration


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
            f"Talk id: {talk_id}")
    return not_split_wav_file


def join_manifests(dir_, output_dir, not_split_wav_dir):
    last_name = dir_.parts[-1]
    joined_name = output_dir / Path(last_name + '.manifest')
    manifest = ""
    for elem in dir_.iterdirs():
        if elem.is_file() and elem.suffix == ".manifest":
            joined_text, joined_duration = get_joined_text_and_duration(elem)
            not_split_wav_file = get_corresponding_not_split_wav_file(elem, not_split_wav_dir)
            if manifest:
                manifest += '\n'
            manifest += json.dumps(
                {"audio_filepath": str(not_split_wav_file), "duration": joined_duration, "text": joined_text})
    with joined_name.open('w') as f:
        f.write(manifest)


def main():
    args = get_args()
    for elem in args.split_manifests.iterdir():
        if elem.is_dir() and any([e.endswith('.manifest') for e in os.listdir(elem)]):
            join_manifests(elem, args.output_dir, args.not_split_wav_dir)


if __name__ == "__main__":
    main()
