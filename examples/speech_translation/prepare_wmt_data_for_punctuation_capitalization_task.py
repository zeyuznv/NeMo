import argparse
import logging
import random
import re
from copy import deepcopy
from io import StringIO
from math import ceil
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')


random.seed(42)


EUROPARL_LINE = re.compile(r"^(.+)(ep(?:-[0-9]{2}){3}(?:-[0-9]{3})?)")
NEWS_COMMENTARY_LOCATION_LINE = re.compile(r"^[A-Z0-9 ]+ – ")
WORD = re.compile(r"\W*\b(?:\w+(?:-\w+)*(?:'\w+)?)\b\W*")
STRIP_START = re.compile(r"^\W+")
STRIP_END = re.compile(r"\W+$")
SUPPORTED_CORPUS_TYPES = ["europarl", "news-commentary", "TED", "rapid"]
SENTENCE_ENDINGS = ".?!"


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("input_files", help="List of files with input data", nargs="+", type=Path)
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path to the output dir with dev.txt, train.txt, and test.txt files.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corpus_types",
        "-c",
        help="List of names of WMT corpuses which is used as raw material for creating punctuation capitalization "
        "dataset. Number and order of elements in this list should be equal to the number of elements in `input_files` "
        "list.",
        choices=SUPPORTED_CORPUS_TYPES,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--size",
        "-S",
        help="Number of sequences in the created dataset. This number includes sequences in train, dev, and test "
        "datasets. By default it is equal to the total number of sentences in the input data.",
    )
    parser.add_argument("--dev_size", "-d", help="Number of sequences in dev data.", type=int, default=10 ** 4)
    parser.add_argument("--test_ratio", "-t", help="Percentage of test data.", type=float, default=0.0)
    parser.add_argument(
        "--sequence_length_range",
        "-r",
        help="Minimum and maximum number words in model input sequences. Number of words is sampled "
        "using uniform distribution.",
        type=int,
        nargs=2,
        default=[2, 64],
    )
    parser.add_argument(
        "--percentage_segments_with_intact_sentences",
        "-w",
        help="For any number of words in a segment percentage of segments with whole sentences can not be lower than "
        "`--percentage_segments_with_intact_sentences`. If this condition can not be satisfied together with the "
        "dataset size and uniform distribution of segment lengths in dataset, distribution will not be uniform: "
        "Probably number of long and short segments will be less than number of segments with average length.",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--clean_data_dir",
        "-C",
        help="Path to directory where cleaned input files are saved. If not provided cleaned input files are "
        "not saved.",
        type=Path,
    )
    args = parser.parse_args()
    args.input_files = [x.expanduser() for x in args.input_files]
    if len(args.input_files) != len(args.corpus_types):
        raise ValueError(
            f"Number {len(args.input_files)} of input files {args.input_files} is not equal to the number "
            f"{len(args.corpus_types)} of corpus types {args.corpus_types}."
        )
    args.output_dir = args.output_dir.expanduser()
    args.clean_data_dir = args.clean_data_dir.expanduser()
    return args


def preprocess_europarl(text):
    f = StringIO(text)
    docs = {}
    for i, line in enumerate(f):
        m = EUROPARL_LINE.match(line)
        if m is None:
            raise ValueError(f"Could not match {i} EUROPARL line {repr(line)}")
        text = m.group(1).strip()
        doc = "europarl_" + m.group(2).strip()
        if doc not in docs:
            docs[doc] = [text]
        else:
            docs[doc].append(text)
    return docs


def preprocess_ted(text):
    soup = BeautifulSoup(text)
    result = {}
    for doc in soup.findAll("doc"):
        doc_id = doc["docid"]
        title = doc.find("title").text
        key = "TED_" + doc_id + "._" + title
        text = ''.join([e for e in doc if isinstance(e, NavigableString)]).strip()
        result[key] = [line.strip() for line in text.split('\n') if line.strip()]
    return result


def preprocess_rapid(text):
    soup = BeautifulSoup(text)
    result = {}
    for file in soup.findAll("file"):
        file_id = file["id"]
        file_utterances = []
        for unit in file.findAll("unit"):
            unit_id = unit["id"]
            segment = unit.find("segment")
            source = segment.find("source")
            target = segment.find("target")
            if source['xml:lang'] == "en":
                text = source.text
            elif target["xml:lang"] == "en":
                text = target.text
            else:
                raise ValueError(
                    f"No utterance in English was found in file {file_id} in unit {unit_id}. "
                    f"Source language: {source['lang']}. Target language: {target['lang']}"
                )
            if text[-1] in SENTENCE_ENDINGS:
                file_utterances.append(text)
        if file_utterances:
            result["rapid_file_" + file_id] = file_utterances
    return result


def preprocess_news_commentary(text):
    result = {}
    discussion_text = []
    discussion_count = 0
    line_idx = 0
    for line in StringIO(text):
        line = line.strip()
        if line:
            if line_idx == 1:
                location_string = NEWS_COMMENTARY_LOCATION_LINE.match(line)
                if location_string is not None:
                    line = line[location_string.span()[1] :]
                discussion_text.append(line)
            elif line_idx > 1:
                discussion_text.append(line)
            line_idx += 1
        else:
            result[f"news-commentary_discussion{discussion_count}"] = discussion_text
            discussion_text = []
            discussion_count += 1
            line_idx = 0
    return result


def cut_words(s, start_word, num_words):
    words = WORD.findall(s)
    s = ''.join(words[start_word : start_word + num_words])
    ss = STRIP_START.match(s)
    if ss is not None:
        s = s[ss.span()[1] :]
    se = STRIP_END.match(s)
    if se is not None:
        s = s[: se.span()[0]]
    return s


def add_docs(all_docs, file_docs, file_name):
    for k, v in file_docs.items():
        duplicate = False
        if k in all_docs:
            if v == all_docs[k]:
                duplicate = True
                logging.warning(f"Duplicate document with name {k} in file {file_name}")
            i = 2
            while k + "_" + str(i) in all_docs:
                if v == all_docs[k + "_" + str(i)]:
                    duplicate = True
                    logging.warning(
                        f"Duplicate documents with names {k} and {k + '_' + str(i)}. One of documents is "
                        f"from file {file_name}"
                    )
                i += 1
            k += "_" + str(i)
        if not duplicate:
            all_docs[k] = v


def arrange_sentences_by_number_of_words(docs, sequence_length_range):
    result = {n: [] for n in range(sequence_length_range[0], sequence_length_range[1])}
    for doc_id, doc in docs.items():
        for start_sentence_i, sentence in enumerate(doc):
            for end_sentence_i in range(start_sentence_i + 1, len(doc)):
                n_words = sum([len(doc[i].split()) for i in range(start_sentence_i, end_sentence_i)])
                if n_words >= sequence_length_range[1] or n_words < sequence_length_range[0]:
                    break
                result[n_words].append((doc_id, start_sentence_i, end_sentence_i))
    return result


def select_close_to_uniform_distribution(
    sentences_by_number_of_words, planned_number_of_segments, percentage_of_segments_with_intact_sentences, all_docs
):
    result = []
    remaining_by_docs = {doc_id: set(range(len(s))) for doc_id, s in all_docs.items()}
    number_of_sentences_by_number_of_words = sorted([(len(v), k) for k, v in sentences_by_number_of_words.items()])
    number_of_words_stats = []
    min_number_of_sentences_for_sentence_len = ceil(
        planned_number_of_segments
        / len(sentences_by_number_of_words)
        / 100
        * percentage_of_segments_with_intact_sentences
    )
    for i, (n, len_) in enumerate(number_of_sentences_by_number_of_words):
        if n < min_number_of_sentences_for_sentence_len:
            tmp = sentences_by_number_of_words[len_]
            planned_number_of_segments -= len(tmp) * int(100 / percentage_of_segments_with_intact_sentences)
            min_number_of_sentences_for_sentence_len = ceil(
                planned_number_of_segments
                / (len(sentences_by_number_of_words) - i)
                / 100
                * percentage_of_segments_with_intact_sentences
            )
        else:
            tmp = random.sample(sentences_by_number_of_words[len_], min_number_of_sentences_for_sentence_len)
        result += tmp
        number_of_words_stats.append((len_, len(tmp)))
        for doc_id, start_i, end_i in tmp:
            if doc_id not in remaining_by_docs:
                remaining_by_docs[doc_id] = set()
            remaining_by_docs[doc_id].difference_update(range(start_i, end_i))
    return result, dict(sorted(number_of_words_stats)), remaining_by_docs


def calculate_how_many_remain_to_cut(number_of_words_stats, size, percentage_segments_with_intact_sentences):
    if percentage_segments_with_intact_sentences > 0:
        factor = (100 - percentage_segments_with_intact_sentences) / percentage_segments_with_intact_sentences
        result = {k: ceil(v * factor) for k, v in number_of_words_stats.items()}
    else:
        n = ceil(size / len(number_of_words_stats))
        result = {k: n for k in number_of_words_stats}
    keys = sorted(result.keys(), key=lambda x: -number_of_words_stats[x])
    total = sum(number_of_words_stats.values()) + sum(result.values())
    key_i = 0
    while total > size:
        if result[keys[key_i]] > 0:
            total -= 1
            result[keys[key_i]] -= 1
        key_i = (key_i + 1) % len(keys)
    for k in keys:
        if result[k] == 0:
            del result[k]
    return result


def create_not_whole_sentence_segments(
    all_docs, remaining_by_docs, number_of_words_stats, size, percentage_segments_with_intact_sentences
):
    result = []
    remaining_by_docs = deepcopy(remaining_by_docs)
    yet_to_cut_by_number_of_words = calculate_how_many_remain_to_cut(
        number_of_words_stats, size, percentage_segments_with_intact_sentences
    )
    nw_i = 0
    done = not bool(yet_to_cut_by_number_of_words)
    while not done:
        for doc_id, remaining in remaining_by_docs.items():
            next_sentence_i = -1
            for i in remaining:
                len_ = len(all_docs[doc_id][i].split())
                if i >= next_sentence_i and len_ > 1:
                    shift = random.randint(0, len_ // 2)
                    text = all_docs[doc_id][i]
                    num_words = len(WORD.findall(text))
                    next_sentence_i = i + 1
                    number_of_words = list(yet_to_cut_by_number_of_words.keys())
                    nw_i %= len(number_of_words)
                    while shift + number_of_words[nw_i] + 1 > num_words and next_sentence_i < len(all_docs[doc_id]):
                        text += ' ' + all_docs[doc_id][next_sentence_i]
                        num_words += len(WORD.findall(all_docs[doc_id][next_sentence_i]))
                        next_sentence_i += 1
                    if shift + number_of_words[nw_i] < num_words:
                        if shift + number_of_words[nw_i] == num_words and shift == 0:
                            shift += 1
                        result.append(cut_words(text, shift, number_of_words[nw_i]))
                        yet_to_cut_by_number_of_words[number_of_words[nw_i]] -= 1
                        if yet_to_cut_by_number_of_words[number_of_words[nw_i]] == 0:
                            del yet_to_cut_by_number_of_words[number_of_words[nw_i]]
                            if not yet_to_cut_by_number_of_words:
                                done = True
                                break
                        nw_i += 1
                    else:
                        break
            if done:
                break
        remaining_by_docs = {doc_id: set(range(len(doc))) for doc_id, doc in all_docs.items()}
    assert len(result) == size - sum(number_of_words_stats.values())
    return result


def create_dataset_string(file_docs):
    result = ""
    for doc_id, doc in file_docs.items():
        result += '\n'.join([f'<doc docid="{doc_id}">'] + doc + ['</doc>'])
    return result


def main():
    args = get_args()
    all_docs = {}
    number_of_sentences_in_input = 0
    for corpus_type, file_path in zip(args.corpus_types, args.input_files):
        logging.info(f"Processing file {file_path}..")
        with file_path.open() as f:
            if corpus_type == SUPPORTED_CORPUS_TYPES[0]:
                file_docs = preprocess_europarl(f.read())
            elif corpus_type == SUPPORTED_CORPUS_TYPES[1]:
                file_docs = preprocess_news_commentary(f.read())
            elif corpus_type == SUPPORTED_CORPUS_TYPES[2]:
                file_docs = preprocess_ted(f.read())
            elif corpus_type == SUPPORTED_CORPUS_TYPES[3]:
                file_docs = preprocess_rapid(f.read())
            else:
                raise ValueError(
                    f"Unsupported corpus type '{corpus_type}. Supported corpus types are {SUPPORTED_CORPUS_TYPES}"
                )
            if args.clean_data_dir is not None:
                args.clean_data_dir.mkdir(parents=True, exist_ok=True)
                with (args.clean_data_dir / Path(file_path.name)).open('w') as f:
                    f.write(create_dataset_string(file_docs))
            add_docs(all_docs, file_docs, file_path)
            number_of_sentences_in_input += sum([len(doc) for doc in file_docs.values()])
    if args.size is None:
        args.size = number_of_sentences_in_input
    sentences_by_number_of_words = arrange_sentences_by_number_of_words(all_docs, args.sequence_length_range)
    if (
        sum([len(x) for x in sentences_by_number_of_words.values()])
        < args.size * args.percentage_segments_with_intact_sentences / 100
    ):
        raise ValueError(
            f"Cannot find enough segments consisting of whole sentences to build dataset with {args.size} segments "
            f"and at least {args.percentage_segments_with_intact_sentences}% segments consisting of whole sentences. "
            f"Try to reduce dataset size of parameter `--percentage_segments_with_intact_sentences"
        )
    result, number_of_words_stats, selected_by_docs = select_close_to_uniform_distribution(
        sentences_by_number_of_words, args.size, args.percentage_segments_with_intact_sentences, all_docs
    )
    for i in range(len(result)):
        result[i] = ' '.join(all_docs[result[i][0]][result[i][1] : result[i][2]])
    result += create_not_whole_sentence_segments(
        all_docs, selected_by_docs, number_of_words_stats, args.size, args.percentage_segments_with_intact_sentences,
    )
    random.shuffle(result)
    if args.dev_size > len(result):
        raise ValueError(f"Parameter `--dev_size={args.dev_size}` is less than size of all dataset ({len(result)})")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_size = int(args.size * args.test_ratio / 100)
    if test_size > 0:
        with (args.output_dir / Path("test.txt")).open('w') as f:
            for i in range(args.dev_size):
                f.write(result[i] + '\n')
    if args.dev_size > 0:
        with (args.output_dir / Path("dev.txt")).open('w') as f:
            for i in range(test_size, test_size + args.dev_size):
                f.write(result[i] + '\n')
    with (args.output_dir / Path("train.txt")).open('w') as f:
        for i in range(args.dev_size + test_size, args.size):
            f.write(result[i] + '\n')


if __name__ == "__main__":
    main()
