#!/usr/bin/env python
# coding: utf-8

# In[2]:

import time
import numpy as np
import pyaudio as pa
import os, time
import nemo
import nemo.collections.asr as nemo_asr
    
from scipy.io import wavfile
import librosa
import ipdb
import datetime
from datetime import datetime as datetime_sub

### From speaker_diarize.py
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.models.label_models import ExtractSpeakerEmbeddingsModel
from nemo.collections.asr.parts.mixins.mixins import DiarizationMixin
# from nemo.collections.asr.data.audio_to_label import get_segments_from_slices
from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map, perform_diarization, write_rttm2manifest, get_DER
from nemo.collections.asr.parts.utils.speaker_utils import get_contiguous_stamps, merge_stamps, labels_to_pyannote_object, rttm_to_labels, labels_to_rttmfile
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.parts.utils.nmse_clustering import COSclustering
from nemo.core.config import hydra_runner
from nemo.utils import logging
import hydra
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import copy
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.utils import logging, model_utils
import torch
from torch.utils.data import DataLoader
import math
seed_everything(42)

# global cfg_pointer

# GT_RTTM_DIR="/disk2/datasets/amicorpus_lapel/lapel_files/amicorpus_test_rttm.scp"
# AUDIO_SCP="/disk2/datasets/amicorpus_lapel/lapel_files/amicorpus_test_wav.scp"
# ORACLE_VAD="/disk2/datasets/amicorpus_lapel/lapel_files/oracle_amicorpus_lapel_test_manifest.json"
# SEG_LENGTH="3"
# SEG_SHIFT="1.5"
# SPK_EMBED_MODEL="speakerdiarization_speakernet"
# DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/ami_oracle_vad'
# reco2num="null"

GT_RTTM_DIR="/disk2/scps/rttm_scps/all_callhome_rttm.scp"
AUDIO_SCP="/disk2/scps/audio_scps/all_callhome.scp"
ORACLE_VAD="/disk2/scps/oracle_vad/modified_oracle_callhome_ch109.json"
reco2num='/disk2/datasets/modified_callhome/RTTMS/reco2num.txt'
SEG_LENGTH=1.5
SEG_SHIFT=0.75
SPK_EMBED_MODEL="/home/taejinp/gdrive/model/ecapa_tdnn/ecapa_tdnn.nemo"
DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/online_ch109_oracle_vad'
reco2num=2

overrides = [
f"diarizer.speaker_embeddings.model_path={SPK_EMBED_MODEL}",
f"diarizer.path2groundtruth_rttm_files={GT_RTTM_DIR}",
f"diarizer.paths2audio_files={AUDIO_SCP}",
f"diarizer.out_dir={DIARIZER_OUT_DIR}",
f"diarizer.oracle_num_speakers={reco2num}",
f"diarizer.speaker_embeddings.oracle_vad_manifest={ORACLE_VAD}",
f"diarizer.speaker_embeddings.window_length_in_sec={SEG_LENGTH}",
f"diarizer.speaker_embeddings.shift_length_in_sec={SEG_SHIFT}",
]


hydra.initialize(config_path="conf")

# cfg = hydra.compose("speaker_diarization.yaml")
cfg_diar = hydra.compose(config_name="speaker_diarization.yaml", overrides=overrides)


# @hydra_runner(config_path="conf", config_name="speaker_diarization.yaml")
# def main(cfg_diar):
    # # global cfg_pointer 
    # logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg_diar)}')
    # sd_model = ClusteringDiarizer(cfg=cfg_diar)
    # # ipdb.set_trace()
    # print(cfg_diar)
    # cfg_pointer = cfg_diar
    # return cfg_diar, sd_model
    # # sd_model.diarize()

# # cfg_diar, sd_model = 
# main()

### >>>>
def get_segments_from_slices(slices, sig, slice_length, shift, audio_signal, audio_lengths):
    """create short speech segments from sclices
    Args:
        slices (int): the number of slices to be created
        slice_length (int): the lenghth of each slice
        shift (int): the amount of slice window shift
        sig (FloatTensor): the tensor that contains input signal

    Returns:
        audio_signal (list): list of sliced input signal
    """
    # fix_siglen = True
    fix_siglen = False
    for slice_id in range(slices):
        start_idx = int(slice_id * shift)
        end_idx = int(start_idx + slice_length)
        signal = sig[start_idx:end_idx]
        audio_lengths.append(len(signal))
        if len(signal) < slice_length:
            signal = repeat_signal(signal, len(signal), slice_length)
        audio_signal.append(signal)
        
        # if fix_siglen:
            # audio_lengths.append(len(signal))
        # else:
    return audio_signal, audio_lengths


class OnlineClusteringDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg
        self._out_dir = self._cfg.diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        # init vad model
        self.has_vad_model = False
        self.has_vad_model_to_save = False

        self._speaker_manifest_path = self._cfg.diarizer.speaker_embeddings.oracle_vad_manifest
        self.AUDIO_RTTM_MAP = None
        self.paths2audio_files = self._cfg.diarizer.paths2audio_files
        
        self.paths2session_audio_files = []
        self.all_hypothesis = []
        self.all_reference = []
        # self._extract_embeddings(self._speaker_manifest_path)

        # def _extract_embeddings(self, manifest_file):
        # def _get_online_embedding(self, signal, manifest_file):

        self.out_rttm_dir = None
    def foo(self):
        pass
    
    def prepare_diarization(self, paths2audio_files: List[str] = None, batch_size: int = 1):
        """
        """
        if paths2audio_files:
            self.paths2audio_files = paths2audio_files
        else:
            if self._cfg.diarizer.paths2audio_files is None:
                raise ValueError("Pass path2audio files either through config or to diarize method")
            else:
                self.paths2audio_files = self._cfg.diarizer.paths2audio_files

        if type(self.paths2audio_files) is str and os.path.isfile(self.paths2audio_files):
            paths2audio_files = []
            with open(self.paths2audio_files, 'r') as path2file:
                for audiofile in path2file.readlines():
                    audiofile = audiofile.strip()
                    paths2audio_files.append(audiofile)

        elif type(self.paths2audio_files) in [list, ListConfig]:
            paths2audio_files = list(self.paths2audio_files)

        else:
            raise ValueError("paths2audio_files must be of type list or path to file containing audio files")

        self.paths2session_audio_files= paths2audio_files

        self.AUDIO_RTTM_MAP = audio_rttm_map(paths2audio_files, self._cfg.diarizer.path2groundtruth_rttm_files)

        # self._extract_embeddings(self._speaker_manifest_path)
        self.out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(self.out_rttm_dir, exist_ok=True)

    def getRTTMfromLabels(self, time_stamps, uniq_key, cluster_labels):
        no_references = False
        lines = time_stamps[uniq_key]
        try:
            assert len(cluster_labels) == len(lines)
        except:
            ipdb.set_trace()
        for idx, label in enumerate(cluster_labels):
            tag = 'speaker_' + str(label)
            lines[idx] += tag
        # ipdb.set_trace()
        a = get_contiguous_stamps(lines)
        labels = merge_stamps(a)
        if self.out_rttm_dir:
            labels_to_rttmfile(labels, uniq_key, self.out_rttm_dir)
        hypothesis = labels_to_pyannote_object(labels)
        self.all_hypothesis.append(hypothesis)

        rttm_file = self.AUDIO_RTTM_MAP[uniq_key]['rttm_path']
        if os.path.exists(rttm_file) and not no_references:
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels)
            self.all_reference.append(reference)
        else:
            no_references = True
            self.all_reference = []
        
        return hypothesis, labels


    def evaluate(self, all_reference, all_hypothesis):
        if len(all_reference) and len(all_hypothesis):
            DER, CER, FA, MISS = get_DER(all_reference, all_hypothesis)
            logging.info(
                "Cumulative results of all the files:  \n FA: {:.4f}\t MISS {:.4f}\t \
                    Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                    FA, MISS, DER, CER
                )
            )


osd_model = OnlineClusteringDiarizer(cfg=cfg_diar)
osd_model.prepare_diarization()
# print(OmegaConf.to_yaml(cfg_diar))
SAMPLE_RATE = 16000
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')







# Preserve a copy of the full config
cfg = copy.deepcopy(asr_model._cfg)
print(OmegaConf.to_yaml(cfg))




# Make config overwrite-able
OmegaConf.set_struct(cfg.preprocessor, False)

# some changes for streaming scenario
cfg.preprocessor.dither = 0.0
cfg.preprocessor.pad_to = 0

# spectrogram normalization constants
normalization = {}
normalization['fixed_mean'] = [
     -14.95827016, -12.71798736, -11.76067913, -10.83311182,
     -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
     -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
     -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
     -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
     -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
     -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
     -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
     -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
     -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
     -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
     -10.10687659, -10.14536695, -10.30828702, -10.23542833,
     -10.88546868, -11.31723646, -11.46087382, -11.54877829,
     -11.62400934, -11.92190509, -12.14063815, -11.65130117,
     -11.58308531, -12.22214663, -12.42927197, -12.58039805,
     -13.10098969, -13.14345864, -13.31835645, -14.47345634]
normalization['fixed_std'] = [
     3.81402054, 4.12647781, 4.05007065, 3.87790987,
     3.74721178, 3.68377423, 3.69344,    3.54001005,
     3.59530412, 3.63752368, 3.62826417, 3.56488469,
     3.53740577, 3.68313898, 3.67138151, 3.55707266,
     3.54919572, 3.55721289, 3.56723346, 3.46029304,
     3.44119672, 3.49030548, 3.39328435, 3.28244406,
     3.28001423, 3.26744937, 3.46692348, 3.35378948,
     2.96330901, 2.97663111, 3.04575148, 2.89717604,
     2.95659301, 2.90181116, 2.7111687,  2.93041291,
     2.86647897, 2.73473181, 2.71495654, 2.75543763,
     2.79174615, 2.96076456, 2.57376336, 2.68789782,
     2.90930817, 2.90412004, 2.76187531, 2.89905006,
     2.65896173, 2.81032176, 2.87769857, 2.84665271,
     2.80863137, 2.80707634, 2.83752184, 3.01914511,
     2.92046439, 2.78461139, 2.90034605, 2.94599508,
     2.99099718, 3.0167554,  3.04649716, 2.94116777]

cfg.preprocessor.normalize = normalization

# Disable config overwriting
OmegaConf.set_struct(cfg.preprocessor, True)
asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
# Set model to inference mode
asr_model.eval();
asr_model = asr_model.to(asr_model.device)

# In[14]:



# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32),                torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


# In[15]:


data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)


# In[16]:


# inference method for audio signal (single instance)
def isOverlap(rangeA, rangeB):
    start1, end1 = rangeA
    start2, end2 = rangeB
    return end1 > start2 and end2 > start1

def getOverlapRange(rangeA, rangeB):
    assert isOverlap(rangeA, rangeB)
    return [ max(rangeA[0], rangeB[0]), min(rangeA[1], rangeB[1])]


def getMergedSpeechLabel(label_list_A, label_list_B):
    if label_list_A == [] or label_list_B == []:
        return label_list_A + label_list_B
    elif label_list_A[-1][1] == label_list_B[0][0]:
        # print(f"-------============== Merging label_list_A[-1][1]: {label_list_A[-1]} label_list_B: {label_list_B[0]}")
        merged_range = [label_list_A[-1][0], label_list_B[0][1]]
        label_list_A.pop()
        label_list_B.pop(0)
        return label_list_A + [merged_range] + label_list_B
    else:
        return label_list_A + label_list_B

def getSubRangeList(target_range: List[float], source_list: List) -> List:
    out_range_list = []
    for s_range in source_list:
        # try:
        if isOverlap(s_range, target_range):
            ovl_range = getOverlapRange(s_range, target_range)
            out_range_list.append(ovl_range)
        # except:
            # ipdb.set_trace()
    return out_range_list 

def getVADfromRTTM(rttm_fullpath):
    out_list = []
    with open(rttm_fullpath, 'r') as rttm_stamp:
        rttm_stamp_list = rttm_stamp.readlines()
        for line in rttm_stamp_list:
            stt = float(line.split()[3])
            end = float(line.split()[4]) + stt
            out_list.append([stt, end])
    return out_list


def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(asr_model.device), audio_signal_len.to(asr_model.device)
    log_probs, encoded_len, predictions = model.forward(
        input_signal=audio_signal, input_signal_length=audio_signal_len
    )
    return log_probs

def clean_trans_and_TS(trans, timestamps):
    """
    Removes the spaces in the beginning and the end.
    timestamps need to be changed and synced accordingly.
    """
    assert (len(trans) > 0) and (len(timestamps) > 0)
    assert len(trans) == len(timestamps)

    trans = trans.lstrip()
    diff_L= len(timestamps) - len(trans)
    timestamps = timestamps[diff_L:]
    
    trans = trans.rstrip()
    diff_R = len(timestamps) - len(trans)
    if diff_R > 0:
        timestamps = timestamps[:-1*diff_R]
    return trans, timestamps

def _get_spaces(trans, timestamps):
    trans, timestamps = clean_trans_and_TS(trans, timestamps)
    assert (len(trans) > 0) and (len(timestamps) > 0)
    assert len(trans) == len(timestamps)

    spaces, word_list = [], []
    stt_idx = 0
    for k, s in enumerate(trans):
        if s == ' ':
            spaces.append([timestamps[k], timestamps[k + 1] - 1])
            word_list.append(trans[stt_idx:k])
            stt_idx = k + 1
    if len(trans) > stt_idx and trans[stt_idx] != ' ':
        word_list.append(trans[stt_idx:])
    # ipdb.set_trace()
    return spaces, word_list

def get_word_ts(text, timestamps, end_stamp):
    if text.strip() == '':
        _trans_words, word_timetamps, _spaces = [], [], []
    elif len(text.split()) == 1:
        _trans_words = [text]
        word_timetamps = [[timestamps[0], end_stamp]]
        _spaces = []
    else:
        try:
            _spaces, _trans_words = _get_spaces(text, timestamps)
        except:
            ipdb.set_trace()
        word_timetamps_middle = [[_spaces[k][1], _spaces[k + 1][0]] for k in range(len(_spaces) - 1)]
        word_timetamps = [[timestamps[0], _spaces[0][0]]] + word_timetamps_middle + [[_spaces[-1][1], end_stamp]]
    return _trans_words, word_timetamps, _spaces


def print_time(string_out, speaker, start_point, end_point, params):
    datetime_offset = 16 * 3600
    if float(start_point) > 3600:
        time_str = "%H:%M:%S.%f"
    else:
        time_str = "%M:%S.%f"
    start_point_str = datetime_sub.fromtimestamp(float(start_point) - datetime_offset).strftime(time_str)[:-4]
    end_point_str = datetime_sub.fromtimestamp(float(end_point) - datetime_offset).strftime(time_str)[:-4]
    strd = "\n[{} - {}] {}: ".format(start_point_str, end_point_str, speaker)
    # print(strd, end=" ")
    # ipdb.set_trace()
    return string_out + strd

def print_word(string_out, word, params):
    word = word.strip()
    # print(word, end=" ")
    return string_out + word + " "

def get_timestamp_in_sec(word_ts_stt_end, params):
    stt = round(params['offset'] + word_ts_stt_end[0] * params['time_stride'], params['round_float'])
    end = round(params['offset'] + word_ts_stt_end[1] * params['time_stride'], params['round_float'])
    return stt, end

def get_num_of_spk_from_labels(labels):
    spk_set = [x.split(' ')[-1].strip() for x in labels]
    return len(set(spk_set))

def get_speaker_label_per_word(uniq_id, words, spaces, word_ts_list, labels, dc):
    # ipdb.set_trace()
    params = {'offset': -0.18, 'time_stride': 0.02, 'round_float': 2}
    start_point, end_point, speaker = labels[0].split()
    word_pos, idx = 0, 0
    # ipdb.set_trace()
    DER, FA, MISS, CER = 100*dc['DER'], 100*dc['FA'], 100*dc['MISS'], 100*dc['CER']
    string_out = f'[Session: {uniq_id}, DER: {DER:.2f}%, FA: {FA:.2f}% MISS: {MISS:.2f}% CER: {CER:.2f}%]'
    string_out = print_time(string_out, speaker, start_point, end_point, params)
    for j, word_ts_stt_end in enumerate(word_ts_list):
        # space_stt_end = [word_ts_stt_end[1], word_ts_stt_end[1]] if j == len(spaces) else spaces[j]
        # trans, logits, timestamps = transcript_logits_list[k]

        word_pos = params['offset'] + word_ts_stt_end[0] 
        # print("word_pos: ", word_pos)
        if word_pos < float(end_point):
            string_out = print_word(string_out, words[j], params)
        else:
            idx += 1
            idx = min(idx, len(labels)-1)
            start_point, end_point, speaker = labels[idx].split()
            string_out = print_time(string_out, speaker, start_point, end_point, params)
            string_out = print_word(string_out, words[j], params)

        stt_sec, end_sec = get_timestamp_in_sec(word_ts_stt_end, params)
    # print(string_out) 
    write_txt(f"/home/taejinp/projects/run_time/online_diar_script/online_trans.txt", string_out.strip())
    
    print("\n")

def write_txt(w_path, val):
    with open(w_path, "w") as output:
        output.write(val + '\n')
    return None

def get_partial_ref_labels(pred_labels, ref_labels):
    last_pred_time = float(pred_labels[-1].split()[1])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        if last_pred_time <= start:
            pass
        elif start < last_pred_time <= end:
            label = f"{start} {last_pred_time} {speaker}"
            ref_labels_out.append(label) 
        elif end < last_pred_time:
            ref_labels_out.append(label) 
    return ref_labels_out 

def online_eval_diarization(pred_labels, rttm_file):
    diar_labels = []
    DER_result_dict = {}
    all_hypotheses = []
    all_references = []
    ref_labels_list = []

    if os.path.exists(rttm_file):
        ref_labels = rttm_to_labels(rttm_file)
        ref_labels = get_partial_ref_labels(pred_labels, ref_labels)
        reference = labels_to_pyannote_object(ref_labels)
        all_references.append(reference)
        # ipdb.set_trace()
    else:
        raise ValueError("No reference RTTM file provided.")

    diar_labels.append(pred_labels)

    est_n_spk = get_num_of_spk_from_labels(pred_labels)
    ref_n_spk = get_num_of_spk_from_labels(ref_labels)
    hypothesis = labels_to_pyannote_object(pred_labels)

    all_hypotheses.append(hypothesis)

    DER, CER, FA, MISS, = get_DER(all_references, all_hypotheses)
    logging.info(
        "Cumulative results of all the files:  \n FA: {:.4f}\t MISS {:.4f}\t\
            Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
            FA, MISS, DER, CER
        )
    )
    DER_result_dict = {"DER": DER, "CER": CER, "FA": FA, "MISS": MISS}
    return DER_result_dict

# In[17]:


# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames

class FrameASR:
    def __init__(self, osd_model, model_definition,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        
        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        self.timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            self.timestep_duration *= block['stride'][0] ** block['repeat']
        self.n_timesteps_overlap = int(frame_overlap / self.timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset

        # >>> For diarization
        self.osd_model = osd_model
        self.time_stamps = {}
        self.curr_uniq_key = None
        self.frame_stt = None
        self.frame_end = None
        self.n_initial_buff_len = self.buffer.shape[0] - self.n_frame_len
        self.transcribe_delay = int(self.n_frame_overlap / self.sr) + 1
        self.embed_seg_len = 1.5
        self.embed_seg_hop = 0.75
        self.n_embed_seg_len = int(self.sr * self.embed_seg_len)
        self.n_embed_seg_hop = int(self.sr * self.embed_seg_hop)
        self.n_embed_seg_count = 1 + int((len(self.buffer) - self.n_embed_seg_len)/self.n_embed_seg_hop)
        # self.embed_update_count = max(1, int((self.n_embed_seg_count-1)/2) )
        self.embed_update_count = int(max(1, self.n_frame_len/self.n_embed_seg_hop))
        self.middle_index = int((self.n_embed_seg_count+1)/2)
        self.diar_buffer_frames = 2 + self.embed_update_count
        self.embs_array_list = []
        self.embs_array = None
        self.frame_index = 0
        self.cum_cluster_labels = np.array([])
        self.silence_list = []
        self.cum_spaces_list = []
        self.silence_logit_frame_threshold = 3
        self.spaces = []
        # self.nonspeech_threshold = 20 #minimun width to consider non-speech activity 
        
        self.nonspeech_threshold = 50 #minimun width to consider non-speech activity 
        self.calibration_offset = -0.18
        self.time_stride = self.timestep_duration
        self.overlap_frames_count = int(self.n_frame_overlap/self.sr)
        self.segment_list = []
        self.segment_abs_time_range_list = []
        self.cum_speech_labels = []

        self.frame_start = 0
        self.rttm_file_path = []
        self.word_seq = []
        self.word_ts_seq = []

        self.reset()

    def _match_speaker(self, cluster_labels):
        if len(self.cum_cluster_labels) == 0:
            self.cum_cluster_labels = np.array(cluster_labels)
        else:
            np_cluster_labels = np.array(cluster_labels)
                # min_len = np.min([self.cum_cluster_labels.shape[0], np_cluster_labels.shape[0]])
            min_len = np.min([len(self.cum_cluster_labels), len(np_cluster_labels) ])
            flip = np.inner(self.cum_cluster_labels[:min_len], 1-np_cluster_labels[:min_len])
            org = np.inner(self.cum_cluster_labels[:min_len], np_cluster_labels[:min_len])
            if flip > org:
                self.cum_cluster_labels = list(self.cum_cluster_labels) + list((1 - np_cluster_labels)[min_len:])
            else:
                self.cum_cluster_labels = list(self.cum_cluster_labels) + list(np_cluster_labels[min_len:])

        return self.cum_cluster_labels


    def _get_online_diar_segments(self):
        frame_list = [self.buffer[self.n_embed_seg_hop*i: self.n_embed_seg_len+self.n_embed_seg_hop*i] for i in range(self.n_embed_seg_count)]
        audio_signal_lens = torch.from_numpy(np.array([self.n_embed_seg_len for k in range(self.n_embed_seg_count)])).to(asr_model.device)
        audio_signal = torch.from_numpy(np.stack(frame_list)).to(asr_model.device)
        return audio_signal, audio_signal_lens

    def _convert_to_torch_var(self, audio_signal):
        # frame_list = [self.buffer[self.n_embed_seg_hop*i: self.n_embed_seg_len+self.n_embed_seg_hop*i] for i in range(self.n_embed_seg_count)]
        # audio_signal_lens = torch.from_numpy(np.array(audio_signal_lens)).to(asr_model.device)
        # try:
        # audio_signal = torch.from_numpy(np.stack(audio_signal)/32768).to(asr_model.device)
        # audio_signal = torch.from_numpy(np.stack(audio_signal)).to(asr_model.device)/32768
        # audio_signal = torch.stack(audio_signal).to(asr_model.device)
        audio_signal = torch.stack(audio_signal).float().to(asr_model.device)
        # audio_signal = torch.div(torch.stack(audio_signal), 32768).float().to(asr_model.device)
        # audio_signal = audio_signal / 32768
        # except:
            # ipdb.set_trace()
        audio_signal_lens = torch.from_numpy(np.array([self.n_embed_seg_len for k in range(audio_signal.shape[0])])).to(asr_model.device)
        return audio_signal, audio_signal_lens

    def _get_absolute_frame_time(self):
        frame_stt = self.frame_index - self.transcribe_delay
        frame_end = self.frame_index + 1 - self.transcribe_delay
        return frame_stt, frame_end 


    def _online_diarization(self, audio_signal, segment_ranges):
        # audio_signal, audio_signal_lens = self._get_online_diar_segments()
        torch_audio_signal, torch_audio_signal_lens = self._convert_to_torch_var(audio_signal)

        try:
            _, embs = self.osd_model._speaker_model.forward(input_signal=torch_audio_signal, input_signal_length=torch_audio_signal_lens)
        except:
            ipdb.set_trace()
        
        self.frame_stt, self.frame_end = self._get_absolute_frame_time() 
        
        self.embs_array = embs.cpu().numpy()
        # try:
        cluster_labels = COSclustering(
            None, self.embs_array, oracle_num_speakers=2, max_num_speaker=2, cuda=True,
        )
        # except: 
            # ipdb.set_trace()
        self.cum_cluster_labels = self._match_speaker(list(cluster_labels))
       
        rl = segment_ranges
        assert len(cluster_labels) == len(rl)
        lines = []
        for idx, label in enumerate(cluster_labels):
            tag = 'speaker_' + str(label)
            lines.append(f"{rl[idx][0]} {rl[idx][1]} {tag}")
        a = get_contiguous_stamps(lines)
        labels = merge_stamps(a)
        # if self.out_rttm_dir:
            # labels_to_rttmfile(labels, uniq_key, self.out_rttm_dir)
        hypothesis = labels_to_pyannote_object(labels)
        # ipdb.set_trace()
        # curr_speaker = -1
        return labels
    
    def _get_ASR_based_VAD_timestamps(self, logits):
        # spaces = self._get_silence_timestamps(logits, symbol_idx = 0,  state_symbol='space')
        blanks = self._get_silence_timestamps(logits, symbol_idx = 28, state_symbol='blank')
        non_speech = list(filter(lambda x:x[1] - x[0] > self.nonspeech_threshold, blanks))
        # if not non_speech == []:
        speech_labels = self._get_speech_labels(logits, non_speech)
        return speech_labels

    def _get_silence_timestamps(self, probs, symbol_idx, state_symbol):
        spaces = []
        idx_state = 0
        state = ''
        
        frame_sec_stt, frame_sec_end = self._get_absolute_frame_time() 
        # print(f" start:{frame_sec_stt} end:{frame_sec_end}")

        # if np.argmax(probs[0]) == 0:
        if np.argmax(probs[0]) == symbol_idx:
            state = state_symbol

        for idx in range(1, probs.shape[0]):
            current_char_idx = np.argmax(probs[idx])
            if state == state_symbol and current_char_idx != 0 and current_char_idx != symbol_idx:
                spaces.append([idx_state, idx-1])
                state = ''
            if state == '':
                if current_char_idx == symbol_idx:
                    state = state_symbol
                    idx_state = idx

        if state == state_symbol:
            spaces.append([idx_state, len(probs)-1])
       
        return spaces
   
    def _get_speech_labels(self, probs, non_speech, ROUND=2):
        offset_sec = self.frame_index - self.transcribe_delay - self.overlap_frames_count
        frame_offset =  float((offset_sec + self.calibration_offset)/self.time_stride)
        speech_labels = []
        
        if non_speech == []: 
            start = (0 + frame_offset)*self.time_stride
            end = (len(probs) -1 + frame_offset)*self.time_stride
            # speech_labels = ["{:.3f} {:.3f} speech".format(start,end)]
            speech_labels.append([start, end])

        else:
            start = frame_offset * self.time_stride
            first_end = (non_speech[0][0]+frame_offset)*self.time_stride
            # speech_labels = ["{:.3f} {:.3f} speech".format(start, first_end)]
            start, first_end = round(start, ROUND), round(first_end, ROUND)
            speech_labels.append([start, first_end])
            if len(non_speech) > 1:
                for idx in range(len(non_speech)-1):
                    start = (non_speech[idx][1] + frame_offset)*self.time_stride
                    end = (non_speech[idx+1][0] + frame_offset)*self.time_stride
                    # speech_labels.append("{:.3f} {:.3f} speech".format(start,end))
                    speech_labels.append([start, end])
            
            last_start = (non_speech[-1][1] + frame_offset)*self.time_stride
            last_end = (len(probs) -1 + frame_offset)*self.time_stride
            # speech_labels.append("{:.3f} {:.3f} speech".format(last_start, last_end))

            last_start, last_end = round(last_start, ROUND), round(last_end, ROUND)
            speech_labels.append([last_start, last_end])

        return speech_labels


    def _decode(self, frame, offset=0):
        torch.manual_seed(0)
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = copy.deepcopy(self.buffer[self.n_frame_len:])
        self.buffer[-self.n_frame_len:] = copy.deepcopy(frame)
       
        logits = infer_signal(asr_model, self.buffer).cpu().numpy()[0]

        speech_labels_from_logits = self._get_ASR_based_VAD_timestamps(logits)
       
        self.frame_stt, self.frame_end = self._get_absolute_frame_time() 
            
        self.buffer_start, audio_signal, audio_lengths = self._get_diar_segments(speech_labels_from_logits)
        # self.buffer = copy.deepcopy(self.signal)
        # self.buffer_start, audio_signal, audio_lengths = self._get_diar_offline_segments(self.uniq_id)
        if self.buffer_start >= 0:
            labels = self._online_diarization(audio_signal, audio_lengths)
        else:
            labels = []
        # ipdb.set_trace()

        curr_speaker = 1
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        whole_buffer_text = self._greedy_decoder(
            logits,
            self.vocab
        )
        self.frame_index += 1
        return decoded[:len(decoded)-offset], curr_speaker, whole_buffer_text, labels
    
    def _get_diar_offline_segments(self, uniq_id, ROUND=2):
        # buffer_start = round(self.frame_stt + self.calibration_offset - float(self.overlap_frames_count), ROUND)
        buffer_start = 0.0
        rttm_file_path = f"/home/taejinp/projects/NeMo/scripts/speaker_recognition/asr_based_diar/oracle_vad_saved/{uniq_id}.rttm"
        speech_labels = getVADfromRTTM(rttm_file_path)
        source_buffer = copy.deepcopy(self.signal)
        sigs_list, sig_rangel_list = self._get_segments_from_buffer(buffer_start, 
                                                                    speech_labels,
                                                                    source_buffer)
        # sigs_list = sigs_list[:50]
        # sig_rangel_list = sig_rangel_list[:50]

        return buffer_start, sigs_list, sig_rangel_list


    def _get_diar_segments(self, speech_labels_from_logits, ROUND=2):
        buffer_start = round(self.frame_stt + self.calibration_offset - float(self.overlap_frames_count), ROUND)
        if buffer_start > 0:
            new_start_abs_sec, buffer_end = self._get_update_abs_time(buffer_start)
            frame_start = round(buffer_start + int(self.n_frame_overlap/self.sr), ROUND)
            self.frame_start = frame_start
            frame_end = frame_start + self.frame_len 
            
            if self.segment_list == []:
                speech_labels_initial = [[round(buffer_start, ROUND),  round(buffer_end, ROUND)]]
                # source_buffer = copy.deepcopy(self.signal)
                source_buffer = copy.deepcopy(self.buffer)
                sigs_list, sig_rangel_list = self._get_segments_from_buffer(buffer_start, 
                                                                            speech_labels_initial, 
                                                                            source_buffer)
                self.segment_list, self.segment_abs_time_range_list = sigs_list, sig_rangel_list
                self.cum_speech_labels = getSubRangeList(target_range=[buffer_start, frame_end], 
                                                         source_list=speech_labels_initial)
            else: 
                # Remove the old segments 
                # Initialize the frame_start as the end of the last range 
                # frame_start = self.segment_abs_time_range_list[-1][1]
                new_start_abs_sec = frame_start
                while True:
                    t_range = self.segment_abs_time_range_list[-1]

                    mid = np.mean(t_range)
                    if frame_start <= t_range[1]:
                        self.segment_abs_time_range_list.pop()
                        self.segment_list.pop()
                        new_start_abs_sec = t_range[0]
                    else:
                        break
                speech_labels_for_update = self._get_speech_labels_for_update(buffer_start, 
                                                                              buffer_end, 
                                                                              frame_start,
                                                                              speech_labels_from_logits,
                                                                              new_start_abs_sec)

                # source_buffer = copy.deepcopy(self.signal)
                source_buffer = copy.deepcopy(self.buffer)
                sigs_list, sig_rangel_list = self._get_segments_from_buffer(buffer_start, 
                                                                            speech_labels_for_update, 
                                                                            source_buffer)

                # Replace with the newly accepted segments.
                self.segment_list.extend(sigs_list)
                self.segment_abs_time_range_list.extend(sig_rangel_list)

                # if len(self.cum_speech_labels) > 1:
                    # print(" ======== Speech labels list: ", speech_labels_from_logits)
                    # print(" ======== Speech labels update list: ", speech_labels_for_update)
                    # print(" ==== self.segment_abs_time_range_list: ", self.segment_abs_time_range_list)
                    # print(" ====== self.cum_speech_labels: ", self.cum_speech_labels)
                    # ipdb.set_trace()
                audio_signal = self.segment_list
                audio_lengths = self.segment_abs_time_range_list

        return buffer_start, self.segment_list, self.segment_abs_time_range_list

    def _get_update_abs_time(self, buffer_start):
        new_bufflen_sec = self.n_frame_len / self.sr
        n_buffer_samples = int(len(self.buffer)/self.sr)
        total_buffer_len_sec = n_buffer_samples/self.frame_len
        buffer_end = buffer_start + total_buffer_len_sec
        # ipdb.set_trace()
        return (buffer_end - new_bufflen_sec), buffer_end

    def _get_speech_labels_for_update(self, buffer_start, buffer_end, frame_start, speech_labels_from_logits, new_start_abs_sec):
        """
        Bring the new speech labels from the current buffer. Then
        1. Concatenate the old speech labels from self.cum_speech_labels for the overlapped region.
            - This goes to new_speech_labels
        2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cum_speech_labels.
        3. Return the speech label from new_start_abs_sec to buffer end.

        """
        new_speech_labels = []
        current_range = [frame_start, frame_start + self.frame_len]
        new_coming_range = [frame_start, buffer_end]
        cursor_to_buffer_end_range = [frame_start, buffer_end]
        update_overlap_range = [new_start_abs_sec, frame_start]
        
        print("_get_speech_labels_for_update: got, new_start_abs_sec :", new_start_abs_sec)

        new_coming_speech_labels = getSubRangeList(target_range=new_coming_range, 
                                                   source_list=speech_labels_from_logits)

        update_overlap_speech_labels = getSubRangeList(target_range=update_overlap_range, 
                                                       source_list=self.cum_speech_labels)
        
        speech_label_for_new_segments = getMergedSpeechLabel(update_overlap_speech_labels, new_coming_speech_labels) 
        
        current_frame_speech_labels = getSubRangeList(target_range=current_range, 
                                                      source_list=speech_labels_from_logits)

        self.cum_speech_labels = getMergedSpeechLabel(self.cum_speech_labels, current_frame_speech_labels) 
        # if len(self.cum_speech_labels) > 1:
            # print(" ========== self.cum_speech_labels: ", self.cum_speech_labels)
            # ipdb.set_trace()
        return speech_label_for_new_segments

    def _get_segments_from_buffer(self, buffer_start, speech_labels_for_update, source_buffer, ROUND=3):
        sigs_list = []
        sig_rangel_list = []
        n_seglen_samples = int(self.embed_seg_len*self.sr)
        n_seghop_samples = int(self.embed_seg_hop*self.sr)
        
        for idx, range_t in enumerate(speech_labels_for_update):
            sigs, sig_lens = [], []
            stt_b = int((range_t[0] - buffer_start) * self.sr)
            end_b = int((range_t[1] - buffer_start) * self.sr)
            # stt_b = int((range_t[0] - 0.0) * self.sr)
            # end_b = int((range_t[1] - 0.0) * self.sr)
            n_dur_samples = int(end_b - stt_b)
            base = math.ceil((n_dur_samples - n_seglen_samples) / n_seghop_samples)
            slices = 1 if base < 0 else base + 1
            # print(f" stt_b: {stt_b}, end_b: {end_b} ")
            try:
                sigs, sig_lens = get_segments_from_slices(slices, 
                                                          torch.from_numpy(source_buffer[stt_b:end_b]),
                                                          n_seglen_samples,
                                                          n_seghop_samples, 
                                                          sigs, 
                                                          sig_lens)
            except:
                ipdb.set_trace()
                continue

            sigs_list.extend(sigs)
            segment_offset = range_t[0]
            
            for seg_idx, sig_len in enumerate(sig_lens):
                seg_len_sec = float(sig_len / self.sr)
                start_abs_sec = round(float(segment_offset + seg_idx*self.embed_seg_hop), ROUND)
                end_abs_sec = round(float(segment_offset + seg_idx*self.embed_seg_hop + seg_len_sec), ROUND)
                sig_rangel_list.append([start_abs_sec, end_abs_sec])

        # print("+++++ speech_labels_for_update:", speech_labels_for_update)
        # print("+++++ sig_rangel_list:", sig_rangel_list)
        # if len(speech_labels_for_update) > 1:
            # ipdb.set_trace()
        return sigs_list, sig_rangel_list

    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged, curr_speaker, whole_buffer_text, diar_labels = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        # text, char_ts, end_stamp = self.greedy_merge_with_ts(whole_buffer_text, self.buffer_start)
        text, char_ts, end_stamp = self.greedy_merge_with_ts(unmerged, self.frame_start)
        return self.greedy_merge(unmerged), curr_speaker, unmerged, whole_buffer_text, text, char_ts, end_stamp, diar_labels

    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s
    
    def greedy_merge_with_ts(self, s, buffer_start, ROUND=2):
        s_merged = ''
        char_ts = [] 
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
                    char_ts.append(round(buffer_start + i*self.time_stride, 2))
        end_stamp = buffer_start + len(s)*self.time_stride
        return s_merged, char_ts, end_stamp

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged


# # Streaming Inference
# 
# Streaming inference depends on a few factors, such as the frame length and buffer size. Experiment with a few values to see their effects in the below cells.

# In[18]:

FRAME_LEN = 1.0
# number of audio channels (expect mono signal)
CHANNELS = 1

CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)

asr = FrameASR(osd_model,
               model_definition = {
                   'sample_rate': SAMPLE_RATE,
                   'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
                   'JasperEncoder': cfg.encoder,
                   'labels': cfg.decoder.vocabulary
               },
               frame_len=FRAME_LEN, frame_overlap=2, 
               offset=4)


# def streamingASRandDIAR(asr=asr, FRAME_LEN=1.0, CHANNELS=1, CHUNK_SIZE=int(FRAME_LEN*SAMPLE_RATE)):
# sdata = sdata[120*samplerate:]
asr.reset()
torch.manual_seed(4)

online_simulation=True
# online_simulation=False

if online_simulation:
    def callback_sim(uniq_key, buffer_counter, sdata, frame_count, time_info, status):
        global empty_counter
        # sdata = sdata/32768
        sampled_seg_sig = sdata[CHUNK_SIZE*(buffer_counter-1):CHUNK_SIZE*(buffer_counter)]
        buffer_counter += 1
        asr.uniq_id = uniq_key
        asr.signal = sdata
        text, curr_speaker, unmerged, whole_buffer_text, text, timestamps, end_stamp, diar_labels = asr.transcribe(sampled_seg_sig)
        # ipdb.set_trace() 
        if asr.buffer_start >= 0:
            _trans_words, word_timetamps, spaces = get_word_ts(text, timestamps, end_stamp)
            assert len(_trans_words) == len(word_timetamps)
            asr.word_seq.extend(_trans_words)
            asr.word_ts_seq.extend(word_timetamps)
            # ipdb.set_trace()
            dc = online_eval_diarization(diar_labels, asr.rttm_file_path)
            get_speaker_label_per_word(asr.uniq_id, asr.word_seq, spaces, asr.word_ts_seq, diar_labels, dc)
            time.sleep(0.9)
            # ipdb.set_trace()
        str_stt_buff = datetime.timedelta(seconds=asr.frame_stt + asr.calibration_offset - float(asr.overlap_frames_count))
        str_end_buff = datetime.timedelta(seconds=asr.frame_end + asr.calibration_offset+ float(asr.overlap_frames_count))

        # if len(text) > 0 and text[-1] == " ":
            # print("")
            # print(f" [{str_m} speaker {curr_speaker}]", raw_unmerged)
        
        # if len(text):
        if True:
            # print(text, end='')
            # print(f" [{str_m} speaker {curr_speaker}]", text)
            # print(f" [{uniq_key}|{str_stt}~{str_end} speaker {curr_speaker}]", whole_buffer_text)
            # print(f" [{uniq_key}|{str_stt_buff}~{str_end_buff} speaker {curr_speaker}]", whole_buffer_text)
            pass
            # print(f" [{uniq_key}|{str_stt}~{str_end} speaker {curr_speaker}]", unmerged)
            # ipdb.set_trace()
            empty_counter = asr.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ',end='')

    frame_count = None
    time_info = None
    status = None

    # session_name = "en_6825"
    session_name = "en_4844"
    # session_name = "en_4521" ### Error
    # session_name = "en_5166"
    
    # samplerate, sdata = wavfile.read(
    # '/disk2/datasets/modified_callhome/callhome_16k/en_6785.wav' 
    # )
    for uniq_key, dcont in osd_model.AUDIO_RTTM_MAP.items():
        if uniq_key == session_name:
            empty_counter = 0
            print(f">>>>>>>>>>>>>>>> Reading uniq_key : {uniq_key}")
            # ipdb.set_trace() 
            samplerate, sdata = wavfile.read(dcont['audio_path'])
            # sdata, samplerate = librosa.load(dcont['audio_path'])
            # ipdb.set_trace()
            # sdata = sdata/32768.0
            asr.curr_uniq_key = uniq_key
            asr.rttm_file_path = dcont['rttm_path']
            asr.time_stamps[asr.curr_uniq_key] = []
            for i in range(int(np.floor(sdata.shape[0]/asr.n_frame_len))):
                callback_sim(uniq_key, i, sdata, frame_count, time_info, status)

else:
    # samplerate, sdata = wavfile.read(
    # '/disk2/datasets/modified_callhome/callhome_16k/en_6785.wav' 
    # )
    # asr.rttm_file = '/disk2/datasets/modified_callhome/RTTMS/ch109/en_6785.rttm'
    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(i, dev.get('name'))

    if len(input_devices):
        dev_idx = -2
        while dev_idx not in input_devices:
            # print('Please type input device ID:')
            dev_idx = 1

        empty_counter = 0
        buffer_counter = 0

        def callback(in_data, frame_count, time_info, status):
            global empty_counter
            global buffer_counter
            global sdata
            global rttm_file
            # signal = np.frombuffer(in_data, dtype=np.int16)
            signal = sdata[CHUNK_SIZE*(buffer_counter):CHUNK_SIZE*(buffer_counter+1)]

            buffer_counter += 1
            # signal = signal/np.max(signal)
            # print("signal:", buffer_counter, np.shape(signal), signal)
            text, curr_speaker, unmerged, whole_buffer_text = asr.transcribe(signal)
            str_m = datetime.timedelta(seconds=(asr.frame_index+1-asr.transcribe_delay))
            # ipdb.set_trace()
            if len(text) > 0 and text[-1] == " ":
                print("")
                # print(f" [{str_m} speaker {curr_speaker}]", raw_unmerged)
            
            if len(text):
                # print(text, end='')
                # print(f" [{str_m} speaker {curr_speaker}]", text)
                print(f" [{str_m} speaker {curr_speaker}]", unmerged)
                empty_counter = asr.offset
            elif empty_counter > 0:
                empty_counter -= 1
                if empty_counter == 0:
                    all_reference, all_hypothesisprint(' ',end='')

            return (in_data, pa.paContinue)

        stream = p.open(format=pa.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=dev_idx,
                        stream_callback=callback,
                        frames_per_buffer=CHUNK_SIZE)
        print('Listening...')
        stream.start_stream()
        # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
        try:
            while stream.is_active():
                time.sleep(0.1)
        finally:        
            stream.stop_stream()
            stream.close()
            p.terminate()
            print()
            print("PyAudio stopped")
    else:
        print('ERROR: No audio input device found.')




# streamingASRandDIAR()


# '/disk2/datasets/modified_callhome/callhome_16k/en_4829.wav'
# '/disk2/datasets/modified_callhome/callhome_16k/en_5888.wav'
# '/disk2/datasets/modified_callhome/callhome_16k/en_6861.wav'
# '/disk2/datasets/modified_callhome/callhome_16k/en_6825.wav' ### MaleFemale
# '/disk2/datasets/modified_callhome/callhome_16k/en_6785.wav' 
# '/disk2/datasets/modified_callhome/callhome_16k/en_6521.wav' ### MaleFemale
# '/disk2/datasets/modified_callhome/callhome_16k/en_6625.wav' ### MaleFemale
# '/disk2/datasets/amicorpus_lapel/amicorpus/EN2004b/audio/EN2004b.Mix-Lapel.wav'
# '/home/taejinp/projects/sample_wav/spkr3.wav'

