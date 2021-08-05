import re
from argparse import ArgumentParser
from pathlib import Path

import yaml


ADD_COLON_RE = re.compile("(^[\\w\\.\\-]+|(?<=.\n|  )[\\w\\.\\-]+)(?=\n | [0-9])")


ORDER = [
    "QuartzNet15x5Base-En",
    "stt_en_jasper10x5dr",
    "stt_en_citrinet_256",
    "stt_en_citrinet_512",
    "stt_en_citrinet_1024",
    "stt_en_citrinet_256_gamma_0_25",
    "stt_en_citrinet_512_gamma_0_25",
    "stt_en_citrinet_1024_gamma_0_25",
    "CitriNet-1024-8x-Stride-Gamma-0.25.nemo",
    "Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-2.0_no_weight_decay_e250-averaged.nemo",
    "stt_en_conformer_ctc_small",
    "stt_en_conformer_ctc_medium",
    "stt_en_conformer_ctc_large",
    "stt_en_conformer_ctc_small_ls",
    "stt_en_conformer_ctc_medium_ls",
    "stt_en_conformer_ctc_large_ls",
    "sel_jarvisasrset_d512_adamwlr2_wd0_aug10x0.05_sp128_500e-last.nemo",
]


def get_args():
    parser = ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--order", "-o", type=Path, default=ORDER)
    args = parser.parse_args()
    if isinstance(args.order, Path):
        args.order = args.order.expanduser()
        order = []
        with args.order.open() as f:
            for line in f:
                order.append(line.strip())
        args.order = order
    args.input = args.input.expanduser()
    return args


def print_data(data, order):
    values = list(data.values())
    if isinstance(values[0], float):
        for key in order:
            if f"{key}.txt" in data or key in data:
                print(key)
        for key in order:
            if f"{key}.txt" in data or key in data:
                if key in data:
                    print(data[key])
                else:
                    print(data[f"{key}.txt"])
    else:
        for k, v in data.items():
            print(f'{k}:')
            print_data(v, order)


def main():
    args = get_args()
    with args.input.open() as f:
        text = f.read()
    text = ADD_COLON_RE.sub(r'\1:', text)
    data = yaml.safe_load(text)
    print_data(data, args.order)


if __name__ == "__main__":
    main()
