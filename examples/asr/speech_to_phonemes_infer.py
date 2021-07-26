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
This script is to be used for bootstrapping a G2P dataset with an ASR model.
"""

from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.metrics.per import PER
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModelPhoneme
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--per_tolerance", type=float, default=1.0, help="used by test")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModelPhoneme.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModelPhoneme.from_pretrained(model_name=args.asr_model)

    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.preprocessor.featurizer.dither = 0.0

    asr_model.setup_test_data(
        test_data_config=OmegaConf.create(
            {'manifest_filepath': args.dataset, 'sample_rate': 16000, 'batch_size': args.batch_size, 'shuffle': False,}
        )
    )
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()
    labels_map = dict([(i, asr_model.decoder.vocabulary[i] + ' ') for i in range(len(asr_model.decoder.vocabulary))])
    per = PER(tokenizer=asr_model.tokenizer)
    hypotheses = []
    references = []
    for test_batch in asr_model.test_dataloader():
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        hypotheses += per.ctc_decoder_predictions_tensor(greedy_predictions)
        for batch_ind in range(greedy_predictions.shape[0]):
            seq_len = test_batch[3][batch_ind].cpu().detach().numpy()
            seq_ids = test_batch[2][batch_ind].cpu().detach().numpy()
            reference = ''.join([labels_map[c] for c in seq_ids[0:seq_len]])
            references.append(reference)
        del test_batch

    per_value = word_error_rate(hypotheses=hypotheses, references=references)
    if per_value > args.per_tolerance:
        raise ValueError(f"got per of {per_value}. it was higher than {args.per_tolerance}")
    logging.info(f'Got PER of {per_value}. Tolerance was {args.per_tolerance}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
