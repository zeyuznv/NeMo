# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script is used to evaluate predicted phonemes against a partial ground
truth that is missing or has hypothetical entries (e.g. OOV, heteronyms).

All tokens to be removed from the reference text should be indicated with a '?'
symbol, or be sandwiched between ?HET_BEGIN_<word> and ?HET_END_<word>.
"""

import json
import os
import re
from argparse import ArgumentParser

from nemo.collections.asr.metrics.wer import word_error_rate


def get_preds_and_refs(manifest_path):
    """
    Retrieve the predictions, tentative reference phonemes with markup for OOV and heteronyms, and stripped
    reference phonemes (w/o markup).

    Args:
        manifest_path: Path to the manifest output of transcribe_speech.py

    Returns:
        oov_entries: Dict with lists 'preds', 'refs', 'markup_refs', and 'text' of entries that had at least one OOV
            word. These may also have heteronyms, but are separated because OOV words do not have phoneme guesses and
            are left empty in the reference.
        non_oov_entries: Dict with lists 'preds', 'refs', 'markup_refs', and 'text' of entries that have no OOV words.
    """
    # We want to distinguish between these because reference phonemes don't have any guesses for OOV, while they do
    # for heteronyms/homographs.
    oov_preds = []
    oov_refs = []
    oov_stripped_refs = []
    oov_text = []

    non_oov_preds = []
    non_oov_refs = []
    non_oov_stripped_refs = []
    non_oov_text = []

    # Iterate through the prediction/tentative reference pairs
    with open(manifest_path, 'r') as f_in:
        for line in f_in:
            line = json.loads(line)
            pred = line['pred_text']
            ref = line['tentative_phonemes']
            text = line['text']

            # Remove ? tokens
            stripped_ref = ' '.join([tkn for tkn in ref.split() if '?' not in tkn])
            if '?OOV' in ref:
                oov_preds.append(pred)
                oov_refs.append(ref)
                oov_stripped_refs.append(ref)
                oov_text.append(text)
            else:
                non_oov_preds.append(pred)
                non_oov_refs.append(ref)
                non_oov_stripped_refs.append(ref)
                non_oov_text.append(text)

    oov_entries = {'preds': oov_preds, 'refs': oov_stripped_refs, 'markup_refs': oov_refs, 'text': oov_text}
    non_oov_entries = {
        'preds': non_oov_preds,
        'refs': non_oov_stripped_refs,
        'markup_refs': non_oov_refs,
        'text': non_oov_text,
    }

    return oov_entries, non_oov_entries


def write_aligned_comparison(f_out, refs, preds, text, n_per_line):
    """
    Writes the aligned predictions and references to the given output file.
    """
    assert len(refs) == len(preds)

    for i in range(len(refs)):
        f_out.write(f"\n{text[i]}")

        entry_ref = re.sub(r'\?HET_BEGIN_[a-z]*[\s]', '[', refs[i])
        entry_ref = re.sub(r'[\s]\?HET_END_[a-z]*', ']', entry_ref)
        entry_ref = entry_ref.split()
        entry_pred = preds[i].split()

        # Split output based on
        max_length = max(len(entry_ref), len(entry_pred))
        for j in range((max_length - 1) // n_per_line + 1):
            s_ind = j * n_per_line
            e_ind = s_ind + n_per_line
            ref_out = '\t'.join(entry_ref[s_ind:e_ind])
            pred_out = '\t'.join(entry_pred[s_ind:e_ind])
            f_out.write(f"\nRefs:\t\t{ref_out}")
            f_out.write(f"\nPreds:\t{pred_out}")

        f_out.write("\n--------------------")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--manifest_path',
        required=True,
        type=str,
        help="Path to the manifest containing both prediction and reference (output of trancribe_speech.py).",
    )
    parser.add_argument(
        '--out_path',
        required=True,
        type=str,
        help="Path to output file where aligned reference/prediction phonemes will be written.",
    )
    parser.add_argument(
        '--n_per_line',
        required=False,
        default=20,
        type=int,
        help="Number of phonemes before wrapping in the alignment comparison output file.",
    )
    args = parser.parse_args()

    # Check for existence of manifest
    if not os.path.exists(args.manifest_path):
        print(f"ERROR: Manifest path does not exist: {args.manifest_path}")
        exit()

    oov_entries, non_oov_entries = get_preds_and_refs(args.manifest_path)
    print(f"OOV entries: {len(oov_entries['refs'])}\t\tNon-OOV entries: {len(non_oov_entries['refs'])}")

    # Calculate PER of entries with complete references (non-OOV)
    non_oov_per = word_error_rate(hypotheses=non_oov_entries['preds'], references=non_oov_entries['refs'])
    print(f"Non-OOV PER with tentative heteronym phonemes: {non_oov_per}")

    oov_per = word_error_rate(hypotheses=oov_entries['preds'], references=oov_entries['refs'])
    print(f"OOV PER (grain of salt, missing OOV word refs): {oov_per}")

    # Write out more readable comparison file for heteronyms vs. reference
    with open(args.out_path, 'w') as f_out:
        f_out.write(f"========== Non-OOV Entries (PER: {non_oov_per}%) ==========")
        write_aligned_comparison(
            f_out, non_oov_entries['refs'], non_oov_entries['preds'], non_oov_entries['text'], args.n_per_line
        )

        f_out.write(f"\n========== OOV Entries (PER: {oov_per}%) ==========")
        write_aligned_comparison(
            f_out, oov_entries['refs'], oov_entries['preds'], oov_entries['text'], args.n_per_line
        )


if __name__ == '__main__':
    main()
