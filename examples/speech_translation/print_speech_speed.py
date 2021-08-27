import re
from argparse import ArgumentParser
from pathlib import Path

from bs4 import BeautifulSoup
from pydub import AudioSegment


TALK_ID_COMPILED_PATTERN = re.compile(r"[1-9][0-9]*(?=\.wav$)")
SPACE_DEDUP = re.compile(r' +')
SOUNDS_DESCR = re.compile(r'^\([^)]+\)( \([^)]+\))*$')  # (Applause) (Laughter) (Silence):


ORDER = list(
    map(
        str,
        [
            17851,
            19330,
            26946,
            17922,
            26257,
            20101,
            1292,
            13517,
            13195,
            27793,
            13340,
            14439,
            17275,
            27105,
            21017,
            32560,
            25727,
            13316,
            15471,
            26073,
            27383,
            17909,
            20519,
        ],
    )
)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--talk_order", "-r", nargs="+", default=ORDER)
    parser.add_argument("--ground_truth_transcript", "-t", type=Path, required=True)
    parser.add_argument("--audio_dir", "-a", type=Path, required=True)
    args = parser.parse_args()
    args.ground_truth_transcript = args.ground_truth_transcript.expanduser()
    args.audio_dir = args.audio_dir.expanduser()
    return args


def get_talk_id_to_text(text):
    soup = BeautifulSoup(text)
    docs = soup.findAll("doc")
    result = {
        doc["docid"]: SPACE_DEDUP.sub(
            ' ', ' '.join([elem.text for elem in doc.findAll("seg") if not SOUNDS_DESCR.match(elem.text)])
        )
        for doc in docs
    }
    return result


def get_talk_id_to_audio_length(audio_dir):
    result = {}
    for elem in audio_dir.iterdir():
        match = TALK_ID_COMPILED_PATTERN.search(str(elem))
        if elem.is_file() and match is not None:
            audio = AudioSegment.from_wav(elem)
            result[match.group(0)] = audio.duration_seconds
    return result


def main():
    args = get_args()
    with args.ground_truth_transcript.open() as f:
        text = f.read()
    talk_id_to_text = get_talk_id_to_text(text)
    talk_id_to_seconds = get_talk_id_to_audio_length(args.audio_dir)
    for talk_id in args.talk_order:
        print(len(talk_id_to_text[talk_id]) / talk_id_to_seconds[talk_id] * 60)


if __name__ == "__main__":
    main()
