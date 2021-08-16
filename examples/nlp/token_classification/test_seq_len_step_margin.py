import sys
sys.path = ["/home/lab/NeMo"] + sys.path

import json
import re
import string
from argparse import ArgumentParser
from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from nemo.collections.nlp.models import PunctuationCapitalizationModel

from score_punctuation_evaluation import PUNCT_LABELS_TO_NUMBERS, compute_scores


MAX_NUM_SUBTOKENS_IN_INPUT = 8184
MAX_SEQ_LENGTH_KEY = "max_seq_length"


BEST_INIT = {"metric": 0, MAX_SEQ_LENGTH_KEY: None, "margin": None, "step": None}
EMPTY_BEST = {
    "punctuation": {"f1_micro": BEST_INIT.copy(), "f1_macro": BEST_INIT.copy(), "accuracy": BEST_INIT.copy()},
    "capitalization": {"f1": BEST_INIT.copy(), "accuracy": BEST_INIT.copy()}
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--labels", "-b", type=Path, required=True)
    parser.add_argument("--source_text", "-t", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--continue_from", "-c", type=Path)
    parser.add_argument("--max_seq_length", "-l", nargs="+", type=int, default=[16, 32, 48, 64, 92, 128, 256, 512])
    parser.add_argument("--margin", "-m", nargs="+", type=int, default=[0, 1, 2, 4, 8, 12, 16, 24, 32])
    parser.add_argument("--step", "-s", nargs="+", type=int, default=[1, 2, 4, 6, 8, 11, 14, 30, 62, 126, 254, 510])
    args = parser.parse_args()
    args.labels = args.labels.expanduser()
    args.source_text = args.source_text.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def remove_punctuation(word: str):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation.replace("'", '')
    return re.sub('[' + all_punct_marks + ']', '', word)


def text_to_labels(text):
    all_text_labels = ""
    for line in [x for x in text.split('\n') if x]:
        line = line.split()
        text = ''
        labels = ''
        punct = ''.join(PUNCT_LABELS_TO_NUMBERS.keys())
        for word in line:
            label = word[-1] if word[-1] in punct else 'O'
            word = remove_punctuation(word)
            if len(word) > 0:
                if word[0].isupper():
                    label += 'U'
                else:
                    label += 'O'

                word = word.lower()
                text += word + ' '
                labels += label + ' '
        all_text_labels += labels.strip() + '\n'
    return all_text_labels


def plot(data, save_filename):
    save_filename.parent.mkdir(exist_ok=True, parents=True)
    for line_var, line_data in data["lines"].items():
        x, y = zip(*sorted(zip(line_data["x"], line_data["y"]), key=lambda xx: xx[0]))
        plt.plot(x, y, label=f"{data['line_variable']}={line_var}", marker='o')
    plt.grid()
    plt.legend()
    plt.xlabel(data['xlabel'])
    plt.ylabel(data['ylabel'])
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.clf()


def save_all_plots(result, output_dir):
    for task, task_scores in result.items():
        data_for_seq_len_fixed_plots = {}
        for margin, margin_scores in result[task]["margin"].items():
            any_step_value = list(result[task]["margin"][margin]["step"].keys())[0]
            series_names = set(result[task]["margin"][margin]["step"][any_step_value].keys())
            assert MAX_SEQ_LENGTH_KEY in series_names
            metrics = series_names - {MAX_SEQ_LENGTH_KEY}
            for m in metrics:
                data_for_margin_fixed_plot = {
                    "lines": {},
                    "xlabel": MAX_SEQ_LENGTH_KEY,
                    "ylabel": m,
                    "line_variable": "step"
                }
                for step in result[task]["margin"][margin]["step"].keys():
                    for i, msl in enumerate(result[task]["margin"][margin]["step"][step][MAX_SEQ_LENGTH_KEY]):
                        init_value = {
                             "lines": {
                                 step: {
                                     "x": [margin],
                                     "y": [result[task]["margin"][margin]["step"][step][m][i]]
                                 },
                             },
                             "xlabel": "margin",
                             "ylabel": m,
                             "line_variable": "step",
                        }
                        if msl not in data_for_seq_len_fixed_plots:
                            data_for_seq_len_fixed_plots[msl] = {m: init_value}
                        elif m not in data_for_seq_len_fixed_plots[msl]:
                            data_for_seq_len_fixed_plots[msl][m] = init_value
                        elif step not in data_for_seq_len_fixed_plots[msl][m]["lines"]:
                            data_for_seq_len_fixed_plots[msl][m]["lines"][step] = init_value["lines"][step]
                        else:
                            data_for_seq_len_fixed_plots[msl][m]["lines"][step]['x'].append(margin)
                            data_for_seq_len_fixed_plots[msl][m]["lines"][step]['y'].append(
                                result[task]["margin"][margin]["step"][step][m][i])
                    data_for_margin_fixed_plot["lines"][step] = {
                        "x": result[task]["margin"][margin]["step"][step][MAX_SEQ_LENGTH_KEY],
                        "y": result[task]["margin"][margin]["step"][step][m],
                    }
                plot(data_for_margin_fixed_plot, save_filename=output_dir / Path(f"{task}/margin{margin}/{m}.png"))
        for msl, msl_data in data_for_seq_len_fixed_plots.items():
            for metric, metric_data in msl_data.items():
                plot(metric_data, save_filename=output_dir / Path(f"{task}/max_seq_length{msl}/{metric}.png"))


def get_best_metrics_and_parameters(result):
    best = deepcopy(EMPTY_BEST)
    for task, task_result in result.items():
        for margin, margin_result in task_result["margin"].items():
            for step, step_result in margin_result["step"].items():
                series_names = set(step_result.keys())
                metric_names = series_names - {MAX_SEQ_LENGTH_KEY}
                for metric in metric_names:
                    for v, msl in zip(step_result[metric], step_result[MAX_SEQ_LENGTH_KEY]):
                        if v > best[task][metric]["metric"]:
                            best[task][metric]["metric"] = v
                            best[task][metric][MAX_SEQ_LENGTH_KEY] = msl
                            best[task][metric]["step"] = step
                            best[task][metric]["margin"] = margin
    return best


def make_margins_and_steps_integers(result):
    fixed = deepcopy(result)
    for task, task_result in result.items():
        for margin, margin_result in task_result["margin"].items():
            for step, step_result in margin_result["step"].items():
                if isinstance(step, str):
                    del fixed[task]["margin"][margin]["step"][step]
                    fixed[task]["margin"][margin]["step"][int(step)] = step_result
            if isinstance(margin, str):
                backup = fixed[task]["margin"][margin]
                del fixed[task]["margin"][margin]
                fixed[task]["margin"][int(margin)] = backup
    return fixed


def main():
    args = get_args()
    with args.source_text.open() as f:
        texts = [line.strip() for line in f]
    with args.labels.open() as f:
        labels_text = f.read()
    model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
    if args.continue_from is None:
        result = {
            "punctuation": {"margin": {}},
            "capitalization": {"margin": {}}
        }
        best = deepcopy(EMPTY_BEST)
    else:
        with args.continue_from.open() as f:
            result = json.load(f)
            result = make_margins_and_steps_integers(result)
        best = get_best_metrics_and_parameters(result)
    for max_seq_length, margin, step in product(args.max_seq_length, args.margin, args.step):
        dscr = f"max_seq_length={max_seq_length}, margin={margin}, step={step}"
        print(dscr)
        if step > max_seq_length - 2 - 2 * margin:
            print(f"SKIPPING because parameter set {dscr} is impossible")
            continue
        try:
            processed = model.add_punctuation_capitalization(
                texts,
                batch_size=MAX_NUM_SUBTOKENS_IN_INPUT // max_seq_length,
                max_seq_length=max_seq_length,
                margin=margin,
                step=step,
            )
        except ValueError:
            print(f"SKIPPING because parameter set {dscr} is impossible")
            continue
        preds_text = text_to_labels('\n'.join(processed))
        scores = compute_scores(preds_text, labels_text)
        for task, task_scores in scores.items():
            margin_dict = result[task]["margin"]
            if margin not in margin_dict:
                margin_dict[margin] = {"step": {}}
            step_dict = margin_dict[margin]["step"]
            if step not in step_dict:
                step_dict[step] = {metric: [value] for metric, value in scores[task].items()}
                step_dict[step][MAX_SEQ_LENGTH_KEY] = [max_seq_length]
            else:
                step_dict[step][MAX_SEQ_LENGTH_KEY].append(max_seq_length)
                for metric, value in scores[task].items():
                    step_dict[step][metric].append(value)
                    if value > best[task][metric]["metric"]:
                        best[task][metric]["metric"] = value
                        best[task][metric][MAX_SEQ_LENGTH_KEY] = max_seq_length
                        best[task][metric]["step"] = step
                        best[task][metric]["margin"] = margin
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / Path("punctuation_capitalization_scores.json")).open('w') as f:
        json.dump(result, f, indent=2)
    with (args.output_dir / Path("best_metrics_and_parameters.json")).open('w') as f:
        json.dump(best, f, indent=2)
    save_all_plots(result, args.output_dir)


if __name__ == "__main__":
    main()
