import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-wav-file", "-w", required=True, type=Path)
    parser.add_argument("--src-text", "-t", required=True, type=Path)
    parser.add_argument("--output-dir", "-d", required=True, type=Path)
    parser.add_argument("--manifest", "-m", required=True, type=Path)
    args = parser.parse_args()
    args.input_wav_file = args.input_wav_file.expanduser()
    args.src_text = args.src_text.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.manifest = args.manifest.expanduser()


def main():
    args = get_args()



if __name__ == "__main__":
    main()