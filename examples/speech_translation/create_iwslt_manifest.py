import argparse
import contextlib
import json
import re
import wave
from pathlib import Path

from bs4 import BeautifulSoup


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
NOT_TRANSCRIPT_PATTERN = re.compile(r"[^a-z0-9 ']", flags=re.I)
SPACE_DEDUP = re.compile(r' +')
SOUNDS_DESCR = re.compile(r'^\([^)]+\)( \([^)]+\))*$')  # (Applause) (Laughter) (Silence):


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", "-a", required=True, type=Path)
    parser.add_argument("--src-text", "-t", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    args.audio_dir = args.audio_dir.expanduser()
    if not args.audio_dir.is_dir():
        raise ValueError("`audio_dir` parameter has to be a path to directory.")
    args.src_text = args.src_text.expanduser()
    if not args.src_text.is_file():
        raise ValueError("`src_text` parameter has to be a path to a file.")
    args.output = args.output.expanduser()
    return args


def get_wav_files(audio_dir):
    result = {}
    for elem in audio_dir.iterdir():
        m = TALK_ID_COMPILED_PATTERN.search(str(elem))
        if elem.is_file() and m is not None:
            result[m.group(0)] = elem
    return result


def get_talk_id_to_text(src_text):
    with src_text.open() as f:
        text = f.read()
    soup = BeautifulSoup(text)
    docs = soup.findAll("doc")
    result = {
        doc["docid"]: SPACE_DEDUP.sub(
            ' ',
            NOT_TRANSCRIPT_PATTERN.sub(
                ' ', ' '.join([elem.text for elem in doc.findAll("seg") if not SOUNDS_DESCR.match(elem.text)]).lower()
            ),
        )
        for doc in docs
    }
    return result


def get_wav_duration(filepath):
    with contextlib.closing(wave.open(str(filepath), 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
    return frames / float(rate)


def main():
    args = get_args()
    talk_id_to_wav_file = get_wav_files(args.audio_dir)
    talk_id_to_text = get_talk_id_to_text(args.src_text)
    manifest = ""
    if len(talk_id_to_text) != len(talk_id_to_wav_file):
        raise ValueError(
            f"Number of documents described in the XML file {args.src_text} is not equal to the number "
            f"of wav files with talk ids found in the directory {args.audio_dir}.\nNumber of documents in the XML "
            f"file: {len(talk_id_to_text)}, number of wav files: {len(talk_id_to_wav_file)}.\n.wav files: "
            f"{talk_id_to_wav_file}"
        )
    for talk_id, text in talk_id_to_text.items():
        if manifest:
            manifest += '\n'
        filepath = talk_id_to_wav_file[talk_id]
        manifest += json.dumps(
            {"audio_filepath": str(filepath), "offset": 0.0, "duration": get_wav_duration(filepath), "text": text}
        )
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('w') as f:
        f.write(manifest)


if __name__ == "__main__":
    main()
