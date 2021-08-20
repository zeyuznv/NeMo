# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import GLUEDataset
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.parts.utils_funcs import list2str, tensor2list
from nemo.core.classes import typecheck
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['ZeroShotIntentModel']


class ZeroShotIntentModel(NLPModel):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.pooler.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
            Initializes ZeroShotIntent model (it is Bert base model with a classifier head atop).
        """

        # needed to setup validation on multiple datasets
        # MNLI task has two separate dev sets: matched and mismatched
        if not self._is_model_being_restored():
            cfg.validation_ds.ds_item = [
                os.path.join(cfg.dataset.data_dir, 'dev_matched.tsv'),
                os.path.join(cfg.dataset.data_dir, 'dev_mismatched.tsv'),
            ]
            cfg.train_ds.ds_item = os.path.join(cfg.dataset.data_dir, cfg.train_ds.ds_item)
            logging.info(f'Using {cfg.validation_ds.ds_item} for model evaluation.')

        self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)

        # for MNLI dataset there are 3 labels: entailment, neutral, contradiction
        num_labels = 3

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=self.register_artifact('language_model.config_file', cfg.language_model.config_file),
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
            vocab_file=self.register_artifact('tokenizer.vocab_file', cfg.tokenizer.vocab_file),
        )

        # uses [CLS] token for classification (the first token)
        self.pooler = SequenceClassifier(
            hidden_size=self.bert_model.config.hidden_size, num_classes=num_labels, log_softmax=False
        )
        self.loss = CrossEntropyLoss()

    def update_data_dir(self, data_dir: str) -> None:
        """
        Update data directory and get data stats with Data Descriptor
        Weights are later used to setup loss

        Args:
            data_dir: path to data directory
        """
        self._cfg.dataset.data_dir = data_dir
        logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
        self._cfg.validation_ds.ds_item = [
            os.path.join(data_dir, 'dev_matched.tsv'),
            os.path.join(data_dir, 'dev_mismatched.tsv'),
        ]

        self._cfg.train_ds.ds_item = os.path.join(data_dir, 'train.tsv')
        logging.info(f'Using {self._cfg.validation_ds.ds_item} for model evaluation.')

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        output = self.pooler(hidden_states=hidden_states)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, labels = batch
        model_output = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        loss = self.loss(logits=model_output, labels=labels)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, input_type_ids, input_mask, labels = batch
        model_output = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=model_output, labels=labels)

        model_output = torch.argmax(model_output, 1)

        eval_tensors = {'preds': model_output, 'labels': labels}
        return {'val_loss': val_loss, 'eval_tensors': eval_tensors}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds = torch.cat([x['eval_tensors']['preds'] for x in outputs])
        labels = torch.cat([x['eval_tensors']['labels'] for x in outputs])

        all_preds = []
        all_labels = []
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            for ind in range(world_size):
                all_preds.append(torch.empty_like(preds))
                all_labels.append(torch.empty_like(labels))
            torch.distributed.all_gather(all_preds, preds)
            torch.distributed.all_gather(all_labels, labels)
        else:
            all_preds.append(preds)
            all_labels.append(labels)

        tensorboard_logs = {}
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            preds = []
            labels = []
            for p in all_preds:
                preds.extend(tensor2list(p))
            for l in all_labels:
                labels.extend(tensor2list(l))

            tensorboard_logs = accuracy(np.array(preds), np.array(labels))
            val_name = self._validation_names[dataloader_idx].upper()
            logging.info(f'{val_name} evaluation: {tensorboard_logs}')

            # writing labels and predictions to a file in output_dir is specified in the config
            output_dir = self._cfg.output_dir
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f'{self.task_name}_{val_name}.txt')
                logging.info(f'Saving labels and predictions to {filename}')
                with open(filename, 'w') as f:
                    f.write('labels\t' + list2str(labels) + '\n')
                    f.write('preds\t' + list2str(preds) + '\n')

        tensorboard_logs['val_loss'] = avg_loss
        for key in tensorboard_logs:
            self.log(f'{key}', tensorboard_logs[key], prog_bar=True)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        if train_data_config is None:
            train_data_config = self._cfg.train_ds

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_multiple_validation_data(self, val_data_config: Union[DictConfig, Dict] = None):
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds

        return super().setup_multiple_validation_data(val_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        file_name = cfg.ds_item
        if not os.path.exists(file_name):
            raise FileNotFoundError(
                "GLUE datasets not found. For more details on how to get the data, see: "
                "https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e"
            )

        dataset = GLUEDataset(
            file_name=file_name,
            task_name='mnli',
            tokenizer=self.tokenizer,
            max_seq_length=self._cfg.dataset.max_seq_length,
            use_cache=self._cfg.dataset.use_cache,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=self._cfg.dataset.drop_last,
        )

    def predict(
        self, sequence: str, candidate_labels: List[str], hypothesis_template="related to {}", multi_class=False
    ) -> Dict:
        """
        Classify a sequence by the given labels (checking entailment).
        Args:
            sequences: The sequence to classify.
            candidate_labels: The list of possible class labels to classify sequence into.
            hypothesis_template (optional): The template used to turn each label into an NLI-style hypothesis.
                This template must include a {} or similar syntax for the candidate label to be inserted into the template.
                For example, the default_template is "related to {}." with the candidate label "car color",
                this would be fed into the model like "<cls> sequence to classify <sep> related to car color <sep>".
                The default template works well in many cases, but it may be worthwhile to experiment with different
                templates depending on the task setting.
            multi_class: Whether or not multiple candidate labels can be true. If :`False`, the scores are normalized
                such that the sum of the label likelihoods for each sequence is 1. If `True`, the labels are
                considered independent and probabilities are normalized for each candidate by doing a softmax of
                the entailment score vs. the contradiction score.
        Returns:
            The result comes as a dictionary with the following keys:
            - sequence (:`str`) -- The sequence for which this is the output.
            - labels (:`List[str]`) -- The labels sorted by order of likelihood.
            - scores (:`List[float]`) -- The probabilities for each of the labels.
        """
        predicted_intents = []
        predicted_slots = []
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Switch model to evaluation mode
            self.eval()
            self.to(device)

            # Dataset.
            infer_datalayer = self._setup_infer_dataloader(queries, test_ds)

            for batch in infer_datalayer:
                input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask = batch

                intent_logits, slot_logits = self.forward(
                    input_ids=input_ids.to(device),
                    token_type_ids=input_type_ids.to(device),
                    attention_mask=input_mask.to(device),
                )

                # predict intents and slots for these examples
                # intents
                intent_preds = tensor2list(torch.argmax(intent_logits, axis=-1))

                # convert numerical outputs to Intent and Slot labels from the dictionaries
                for intent_num in intent_preds:
                    if intent_num < len(intent_labels):
                        predicted_intents.append(intent_labels[int(intent_num)])
                    else:
                        # should not happen
                        predicted_intents.append("Unknown Intent")

                # slots
                slot_preds = torch.argmax(slot_logits, axis=-1)

                for slot_preds_query, mask_query in zip(slot_preds, subtokens_mask):
                    query_slots = ''
                    for slot, mask in zip(slot_preds_query, mask_query):
                        if mask == 1:
                            if slot < len(slot_labels):
                                query_slots += slot_labels[int(slot)] + ' '
                            else:
                                query_slots += 'Unknown_slot '
                    predicted_slots.append(query_slots.strip())

        finally:
            # set mode back to its original value
            self.train(mode=mode)

        return predicted_intents, predicted_slots

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass


def accuracy(preds, labels):
    if len(preds) != len(labels):
        raise ValueError("Predictions and labels must have the same length")

    return {"acc": (preds == labels).mean()}
