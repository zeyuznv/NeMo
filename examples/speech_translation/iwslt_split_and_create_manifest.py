import argparse
import json
import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from pydub import AudioSegment


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
NOT_TRANSCRIPT_PATTERN = re.compile(r"[^a-z ']")
SPACE_DEDUP = re.compile(r' +')
SOUNDS_DESCR = re.compile(r'^\([^)]+\)$')

NUMBER_OF_MS_IN_1_SEC = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", "-a", required=True, type=Path)
    parser.add_argument("--segmentation-file", "-s", required=True, type=Path)
    parser.add_argument("--src-text", "-t", required=True, type=Path)
    parser.add_argument("--output-dir", "-d", required=True, type=Path)
    args = parser.parse_args()
    args.audio_dir = args.audio_dir.expanduser()
    args.segmentation_file = args.segmentation_file.expanduser()
    args.src_text = args.src_text.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def get_wav_files(audio_dir):
    result = {}
    for elem in audio_dir.iterdir():
        m = TALK_ID_COMPILED_PATTERN.search(str(elem))
        if elem.is_file() and m is not None:
            result[m.group(0)] = elem
    return result


def load_markup(segmentation_file):
    result = {}
    with open(segmentation_file) as f:
        data = yaml.load(f)
    for seg in data:
        m = TALK_ID_COMPILED_PATTERN.search(seg["wav"])
        if m is None:
            raise ValueError(f"Could not extract talk id from seg file name {seg['wav']}")
        talk_id = m.group(0)
        if talk_id not in result:
            result[talk_id] = []
        result[talk_id].append((seg["offset"], seg["duration"]))
    for k, v in result.items():
        result[k] = sorted(v)
    return result


def get_talk_id_to_texts(src_text):
    with src_text.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    docs = soup.findAll("doc")
    result = {
        doc["docid"]:
            [
                SPACE_DEDUP.sub(' ', NOT_TRANSCRIPT_PATTERN.sub(' ', elem.text.lower()))
                for elem in doc.findAll("seg") if not SOUNDS_DESCR.match(elem.text)
            ]
        for doc in docs
    }
    return result


def split_file(talk_id, file_markup, file_path, file_texts, output_dir):
    split_dir = output_dir / Path(talk_id)
    seg_wavs = split_dir / Path("wavs")
    seg_manifest_path = split_dir / Path("manifest")
    audio = AudioSegment.from_wav(file_path)
    manifest = ""
    if len(file_markup) != len(file_texts):
        raise ValueError(f"Number of audio segments and number of texts are not equal for talk {talk_id}")
    for i, (seg_time, text) in enumerate(zip(file_markup, file_texts)):
        audio_seg = audio[seg_time[0] * NUMBER_OF_MS_IN_1_SEC:(seg_time[0] + seg_time[1]) * NUMBER_OF_MS_IN_1_SEC]
        save_path = seg_wavs / Path(str(i) + '.wav')
        if manifest:
            manifest += '\n'
        manifest += json.dumps({"audio_filepath": str(save_path), "duration": audio_seg.duration_seconds, "text": text})
        audio_seg.export(save_path, format="wav")
    with seg_manifest_path.open('w') as f:
        f.write(manifest)


def main():
    args = get_args()
    markup = load_markup(args.segmentation_file)
    wav_paths = get_wav_files(args.audio_dir)
    texts = get_talk_id_to_texts(args.src_texts)
    args.output_dir.makedirs(exist_ok=True, parents=True)
    for talk_id, file_markup in markup.items():
        split_file(talk_id, file_markup, wav_paths[talk_id], texts[talk_id], args.output_dir)


if __name__ == "__main__":
    main()
