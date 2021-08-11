#!/usr/bin/env python
# coding: utf-8

# In[2]:

import time
import numpy as np
import pyaudio as pa
import os, time
import nemo
import nemo.collections.asr as nemo_asr
import soundfile as sf
from pyannote.metrics.diarization import DiarizationErrorRate

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
from nemo.collections.asr.parts.utils.nmse_clustering import (
    NMESC,
    _SpectralClustering,
    COSclustering,
    getCosAffinityMatrix,
    getAffinityGraphMat,
    getLaplacian,
    getLamdaGaplist,
    eigDecompose,
)

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
    
    @staticmethod 
    def estimateNumofSpeakers(affinity_mat, max_num_speaker, is_cuda=False):
        """
        Estimates the number of speakers using eigen decompose on laplacian Matrix.
        affinity_mat: (array)
            NxN affitnity matrix
        max_num_speaker: (int)
            Maximum number of clusters to consider for each session
        is_cuda: (bool)
            if cuda availble eigh decomposition would be computed on GPUs
        """
        laplacian = getLaplacian(affinity_mat)
        lambdas, _ = eigDecompose(laplacian, is_cuda)
        lambdas = np.sort(lambdas)
        lambda_gap_list = getLamdaGaplist(lambdas)
        num_of_spk = np.argmax(lambda_gap_list[: min(max_num_speaker, len(lambda_gap_list))]) + 1
        return num_of_spk, lambdas, lambda_gap_list

    def OnlineCOSclustering(self, key, emb, oracle_num_speakers=None, max_num_speaker=8, min_samples=6, fixed_thres=None, cuda=False):
        """
        Clustering method for speaker diarization based on cosine similarity.

        Parameters:
            key: (str)
                A unique ID for each speaker

            emb: (numpy array)
                Speaker embedding extracted from an embedding extractor

            oracle_num_speaker: (int or None)
                Oracle number of speakers if known else None

            max_num_speaker: (int)
                Maximum number of clusters to consider for each session

            min_samples: (int)
                Minimum number of samples required for NME clustering, this avoids
                zero p_neighbour_lists. Default of 6 is selected since (1/rp_threshold) >= 4
                when max_rp_threshold = 0.25. Thus, NME analysis is skipped for matrices
                smaller than (min_samples)x(min_samples).
        Returns:
            Y: (List[int])
                Speaker label for each segment.
        """
        mat = getCosAffinityMatrix(emb)
        if oracle_num_speakers:
            max_num_speaker = oracle_num_speakers

        nmesc = NMESC(
            mat,
            max_num_speaker=max_num_speaker,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=30,
            fixed_thres=None,
            NME_mat_size=300,
            cuda=cuda,
        )

        if emb.shape[0] > min_samples:
            est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            affinity_mat = mat
            est_num_of_spk, _, _ = self.estimateNumofSpeakers(affinity_mat, max_num_speaker, cuda)

        if oracle_num_speakers:
            est_num_of_spk = oracle_num_speakers

        spectral_model = _SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda)
        Y = spectral_model.predict(affinity_mat)

        return Y

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
    data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.sample_rate)
    data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
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

def match_diar_labels_speakers(old_diar_labels, new_diar_labels):
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True, uem=None)
    reference = labels_to_pyannote_object(old_diar_labels)
    hypothesis = labels_to_pyannote_object(new_diar_labels)
    metric(reference, hypothesis, detailed=True)
    mapping_dict = metric.optimal_mapping(reference, hypothesis)
    return mapping_dict 

def get_mapped_speaker(speaker_mapping, speaker):
    if speaker in speaker_mapping:
        new_speaker = speaker_mapping[speaker]
    else:
        new_speaker = speaker
    return new_speaker

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
        
    return audio_signal, audio_lengths

def get_speaker_label_per_word(uniq_id, words, spaces, word_ts_list, labels, dc):
    params = {'offset': -0.18, 'time_stride': 0.02, 'round_float': 2}
    start_point, end_point, speaker = labels[0].split()
    word_pos, idx = 0, 0
    DER, FA, MISS, CER = 100*dc['DER'], 100*dc['FA'], 100*dc['MISS'], 100*dc['CER']
    string_out = f'[Session: {uniq_id}, DER: {DER:.2f}%, FA: {FA:.2f}% MISS: {MISS:.2f}% CER: {CER:.2f}%]'
    string_out = print_time(string_out, speaker, start_point, end_point, params)
    for j, word_ts_stt_end in enumerate(word_ts_list):
        word_pos = word_ts_stt_end[0] 
        if word_pos < float(end_point):
            string_out = print_word(string_out, words[j], params)
        else:
            idx += 1
            idx = min(idx, len(labels)-1)
            start_point, end_point, speaker = labels[idx].split()
            string_out = print_time(string_out, speaker, start_point, end_point, params)
            string_out = print_word(string_out, words[j], params)

        stt_sec, end_sec = get_timestamp_in_sec(word_ts_stt_end, params)
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

def replace_old_labels(segment_abs_time_range_list):
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

def read_wav(audio_file):
    with sf.SoundFile(audio_file, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read(dtype='float32')
    samples = samples.transpose()
    return sample_rate, samples


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

def load_ASR_model():
    # Preserve a copy of the full config
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')
    cfg = copy.deepcopy(asr_model._cfg)
    print(OmegaConf.to_yaml(cfg))

    # Make config overwrite-able
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0
    # cfg.preprocessor.normalize = normalization

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)
    
    # Set model to inference mode
    asr_model.eval();
    asr_model = asr_model.to(asr_model.device)

    return cfg, asr_model

def callback_sim(uniq_key, buffer_counter, sdata, frame_count, time_info, status):
    asr.buffer_counter = buffer_counter
    sampled_seg_sig = sdata[asr.CHUNK_SIZE*(asr.buffer_counter):asr.CHUNK_SIZE*(asr.buffer_counter+1)]
    asr.uniq_id = uniq_key
    asr.signal = sdata
    text, timestamps, end_stamp, diar_labels = asr.transcribe(sampled_seg_sig)
    
    if asr.buffer_start >= 0 and (diar_labels != [] and diar_labels != None):
        _trans_words, word_timetamps, spaces = get_word_ts(text, timestamps, end_stamp)
        assert len(_trans_words) == len(word_timetamps)
        asr.word_seq.extend(_trans_words)
        asr.word_ts_seq.extend(word_timetamps)
        dc = online_eval_diarization(diar_labels, asr.rttm_file_path)
        get_speaker_label_per_word(asr.uniq_id, asr.word_seq, spaces, asr.word_ts_seq, diar_labels, dc)
        asr.result_diar_labels = diar_labels
        time.sleep(0.1)


class Frame_ASR_DIAR:
    def __init__(self, asr_model, online_diar_model, model_definition,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''

        # >>> For Streaming (Frame) ASR
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
        self.CHUNK_SIZE = int(self.frame_len*self.sr)

        # >>> For diarization
        self.asr_model = asr_model
        self.online_diar_model = online_diar_model
        self.embed_seg_len = 1.5
        self.embed_seg_hop = 0.75
        self.n_embed_seg_len = int(self.sr * self.embed_seg_len)
        self.n_embed_seg_hop = int(self.sr * self.embed_seg_hop)
        
        self.embs_array = None
        self.frame_index = 0
        self.cumulative_cluster_labels = []
        
        self.nonspeech_threshold = 50  #minimun width to consider non-speech activity 
        self.calibration_offset = -0.18
        self.time_stride = self.timestep_duration
        self.overlap_frames_count = int(self.n_frame_overlap/self.sr)
        self.segment_list = []
        self.segment_abs_time_range_list = []
        self.cumulative_speech_labels = []

        self.frame_start = 0
        self.rttm_file_path = []
        self.word_seq = []
        self.word_ts_seq = []
        self.result_diar_labels = []
        self.merged_cluster_labels = []
        self.diar_buffer_length_sec = 120
        self.use_offline_asr = False
        self.offline_logits = None
        self.debug_mode = False
        self.reset()

    def _match_speaker(self, cluster_labels):
        if len(self.cumulative_cluster_labels) == 0:
            self.cumulative_cluster_labels = np.array(cluster_labels)
        else:
            np_cluster_labels = np.array(cluster_labels)
            min_len = np.min([len(self.cumulative_cluster_labels), len(np_cluster_labels) ])
            flip = np.inner(self.cumulative_cluster_labels[:min_len], 1-np_cluster_labels[:min_len])
            org = np.inner(self.cumulative_cluster_labels[:min_len], np_cluster_labels[:min_len])
            if flip > org:
                self.cumulative_cluster_labels = list(self.cumulative_cluster_labels) + list((1 - np_cluster_labels)[min_len:])
            else:
                self.cumulative_cluster_labels = list(self.cumulative_cluster_labels) + list(np_cluster_labels[min_len:])

        return self.cumulative_cluster_labels

    def _convert_to_torch_var(self, audio_signal):
        audio_signal = torch.stack(audio_signal).float().to(self.asr_model.device)
        audio_signal_lens = torch.from_numpy(np.array([self.n_embed_seg_len for k in range(audio_signal.shape[0])])).to(self.asr_model.device)
        return audio_signal, audio_signal_lens

    def _process_cluster_labels(self, segment_ranges, cluster_labels):
        self.cumulative_cluster_labels = list(cluster_labels)
        assert len(cluster_labels) == len(segment_ranges)
        lines = []
        for idx, label in enumerate(cluster_labels):
            tag = 'speaker_' + str(label)
            lines.append(f"{segment_ranges[idx][0]} {segment_ranges[idx][1]} {tag}")
        cont_lines = get_contiguous_stamps(lines)
        string_labels = merge_stamps(cont_lines)
        return string_labels

    def _online_diarization(self, audio_signal, segment_ranges):
        torch_audio_signal, torch_audio_signal_lens = self._convert_to_torch_var(audio_signal)
        _, embs = self.online_diar_model._speaker_model.forward(input_signal=torch_audio_signal, input_signal_length=torch_audio_signal_lens)
        self.embs_array = embs.cpu().numpy()
        
        cluster_labels = self.online_diar_model.OnlineCOSclustering(
            None, self.embs_array, oracle_num_speakers=2, max_num_speaker=2, cuda=True,
        )
        string_labels = self._process_cluster_labels(segment_ranges, cluster_labels)
        updated_labels = self.update_speaker_label_segments(string_labels)
        return updated_labels

    def update_speaker_label_segments(self, labels):
        assert labels != []
        if self.merged_cluster_labels == []:
            self.merged_cluster_labels = copy.deepcopy(labels)
            return labels
        else:
            new_labels = []
            mapping_dict = match_diar_labels_speakers(self.merged_cluster_labels, labels)
            update_start = max([self.buffer_start - self.diar_buffer_length_sec, self.buffer_init_time]) 
            while len(labels) > 0:
                stt_b, end_b, spk_b = labels[-1].split()
                b_range = float(stt_b), float(end_b)
                if update_start < b_range[0] or (b_range[0] <= update_start < b_range[1]):
                    label = labels.pop()
                    stt_str, end_str, spk_str = label.split()
                    spk_str = get_mapped_speaker(mapping_dict, spk_str)
                    new_labels.insert(0, f"{stt_str} {end_str} {spk_str}")
                else:
                    break
            
            while len(self.merged_cluster_labels) > 0:
                stt_a, end_a, spk_a = self.merged_cluster_labels[-1].split()
                a_range = float(stt_a), float(end_a)
                if update_start < a_range[0] or (a_range[0] <= update_start < a_range[1]):
                    self.merged_cluster_labels.pop()
                else:
                    break

            self.merged_cluster_labels.extend(new_labels)
            return self.merged_cluster_labels


    def _get_ASR_based_VAD_timestamps(self, logits, use_offset_time=True):
        blanks = self._get_silence_timestamps(logits, symbol_idx = 28, state_symbol='blank')
        non_speech = list(filter(lambda x:x[1] - x[0] > self.nonspeech_threshold, blanks))
        if use_offset_time:
            offset_sec = int(self.frame_index - 2*self.overlap_frames_count)
        else:
            offset_sec = 0
        speech_labels = self._get_speech_labels(logits, non_speech, offset_sec)
        return speech_labels

    def _get_silence_timestamps(self, probs, symbol_idx, state_symbol):
        spaces = []
        idx_state = 0
        state = ''
        
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
   
    def _get_speech_labels(self, probs, non_speech, offset_sec, ROUND=2):
        frame_offset =  float((offset_sec + self.calibration_offset)/self.time_stride)
        speech_labels = []
        
        if non_speech == []: 
            start = (0 + frame_offset)*self.time_stride
            end = (len(probs) -1 + frame_offset)*self.time_stride
            start, end = round(start, ROUND), round(end, ROUND)
            if start != end:
                speech_labels.append([start, end])

        else:
            start = frame_offset * self.time_stride
            first_end = (non_speech[0][0]+frame_offset)*self.time_stride
            start, first_end = round(start, ROUND), round(first_end, ROUND)
            if start != first_end:
                speech_labels.append([start, first_end])

            if len(non_speech) > 1:
                for idx in range(len(non_speech)-1):
                    start = (non_speech[idx][1] + frame_offset)*self.time_stride
                    end = (non_speech[idx+1][0] + frame_offset)*self.time_stride
                    start, end = round(start, ROUND), round(end, ROUND)
                    if start != end:
                        speech_labels.append([start, end])
            
            last_start = (non_speech[-1][1] + frame_offset)*self.time_stride
            last_end = (len(probs) -1 + frame_offset)*self.time_stride

            last_start, last_end = round(last_start, ROUND), round(last_end, ROUND)
            if last_start != last_end:
                speech_labels.append([last_start, last_end])

        return speech_labels


    def _decode_and_cluster(self, frame, offset=0):
        torch.manual_seed(0)
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = copy.deepcopy(self.buffer[self.n_frame_len:])
        self.buffer[-self.n_frame_len:] = copy.deepcopy(frame)
     
        if self.use_offline_asr:
            logits_start = self.frame_index * int(self.frame_len/self.time_stride)
            logits_end = logits_start + int((2*self.frame_overlap+self.frame_len)/self.time_stride)+1
            logits = self.offline_logits[logits_start:logits_end]
        else:
            logits = infer_signal(asr_model, self.buffer).cpu().numpy()[0]

        speech_labels_from_logits = self._get_ASR_based_VAD_timestamps(logits)
       
        if self.debug_mode:
            self.buffer_start, audio_signal, audio_lengths, speech_labels_used = self._get_diar_offline_segments(self.uniq_id)
        else:
            self.buffer_start, audio_signal, audio_lengths = self._get_diar_segments(speech_labels_from_logits)
        
        if self.buffer_start >= 0:
            labels = self._online_diarization(audio_signal, audio_lengths)
        else:
            labels = []

        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        
        self.frame_index += 1
        unmerged =decoded[:len(decoded)-offset]
        return unmerged, labels
    
    def _get_diar_offline_segments(self, uniq_id, ROUND=2):
        use_oracle_VAD = False
        buffer_start = 0.0
        self.buffer_init_time = buffer_start
        
        if use_oracle_VAD:
            user_folder = "/home/taejinp/projects"
            rttm_file_path = f"{user_folder}/NeMo/scripts/speaker_recognition/asr_based_diar/oracle_vad_saved/{uniq_id}.rttm"
            speech_labels = getVADfromRTTM(rttm_file_path)
        else:
            speech_labels = self._get_ASR_based_VAD_timestamps(self.offline_logits[200:], use_offset_time=False)

        speech_labels[0][0] = 0

        speech_labels = [[round(x, ROUND), round(y, ROUND)] for (x, y) in speech_labels ]
        source_buffer = copy.deepcopy(self.signal)
        sigs_list, sig_rangel_list = self._get_segments_from_buffer(buffer_start, 
                                                                    speech_labels,
                                                                    source_buffer)
        return buffer_start, sigs_list, sig_rangel_list, speech_labels


    def _get_diar_segments(self, speech_labels_from_logits, ROUND=2):
        buffer_start = round(float(self.frame_index - 2*self.overlap_frames_count), ROUND)

        if buffer_start >= 0:
            new_start_abs_sec, buffer_end = self._get_update_abs_time(buffer_start)
            self.frame_start = round(buffer_start + int(self.n_frame_overlap/self.sr), ROUND)
            frame_end = self.frame_start + self.frame_len 
            
            if self.segment_list == []:
                self.buffer_init_time = self.buffer_start
                speech_labels_initial = self._get_speech_labels_for_update(buffer_start, 
                                                                           buffer_end, 
                                                                           self.frame_start,
                                                                           speech_labels_from_logits,
                                                                           new_start_abs_sec)
                source_buffer = copy.deepcopy(self.buffer)
                sigs_list, sig_rangel_list = self._get_segments_from_buffer(buffer_start, 
                                                                            speech_labels_initial, 
                                                                            source_buffer)
                self.segment_list, self.segment_abs_time_range_list = sigs_list, sig_rangel_list
                self.cumulative_speech_labels = getSubRangeList(target_range=[buffer_start, frame_end], 
                                                         source_list=speech_labels_initial)
            else: 
                # Remove the old segments 
                # Initialize the frame_start as the end of the last range 
                # frame_start = self.segment_abs_time_range_list[-1][1]
                new_start_abs_sec = self.frame_start
                while True and len(self.segment_list) > 0:
                    t_range = self.segment_abs_time_range_list[-1]

                    mid = np.mean(t_range)
                    if self.frame_start <= t_range[1]:
                        self.segment_abs_time_range_list.pop()
                        self.segment_list.pop()
                        new_start_abs_sec = t_range[0]
                    else:
                        break
                
                speech_labels_for_update = self._get_speech_labels_for_update(buffer_start, 
                                                                              buffer_end, 
                                                                              self.frame_start,
                                                                              speech_labels_from_logits,
                                                                              new_start_abs_sec)

                source_buffer = copy.deepcopy(self.buffer)
                sigs_list, sig_rangel_list = self._get_segments_from_buffer(buffer_start, 
                                                                            speech_labels_for_update, 
                                                                            source_buffer)

                self.segment_list.extend(sigs_list)
                self.segment_abs_time_range_list.extend(sig_rangel_list)

        return buffer_start, self.segment_list, self.segment_abs_time_range_list

    def _get_update_abs_time(self, buffer_start):
        new_bufflen_sec = self.n_frame_len / self.sr
        n_buffer_samples = int(len(self.buffer)/self.sr)
        total_buffer_len_sec = n_buffer_samples/self.frame_len
        buffer_end = buffer_start + total_buffer_len_sec
        return (buffer_end - new_bufflen_sec), buffer_end

    def _get_speech_labels_for_update(self, buffer_start, buffer_end, frame_start, speech_labels_from_logits, new_start_abs_sec):
        """
        Bring the new speech labels from the current buffer. Then
        1. Concatenate the old speech labels from self.cumulative_speech_labels for the overlapped region.
            - This goes to new_speech_labels
        2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cumulative_speech_labels.
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
                                                       source_list=self.cumulative_speech_labels)
        
        speech_label_for_new_segments = getMergedSpeechLabel(update_overlap_speech_labels, new_coming_speech_labels) 
        
        current_frame_speech_labels = getSubRangeList(target_range=current_range, 
                                                      source_list=speech_labels_from_logits)

        self.cumulative_speech_labels = getMergedSpeechLabel(self.cumulative_speech_labels, current_frame_speech_labels) 
        return speech_label_for_new_segments

    def _get_segments_from_buffer(self, buffer_start, speech_labels_for_update, source_buffer, ROUND=3):
        sigs_list = []
        sig_rangel_list = []
        n_seglen_samples = int(self.embed_seg_len*self.sr)
        n_seghop_samples = int(self.embed_seg_hop*self.sr)
        
        for idx, range_t in enumerate(speech_labels_for_update):
            if range_t[0] < 0:
                continue
            sigs, sig_lens = [], []
            stt_b = int((range_t[0] - buffer_start) * self.sr)
            end_b = int((range_t[1] - buffer_start) * self.sr)
            n_dur_samples = int(end_b - stt_b)
            base = math.ceil((n_dur_samples - n_seglen_samples) / n_seghop_samples)
            slices = 1 if base < 0 else base + 1
            sigs, sig_lens = get_segments_from_slices(slices, 
                                                      torch.from_numpy(source_buffer[stt_b:end_b]),
                                                      n_seglen_samples,
                                                      n_seghop_samples, 
                                                      sigs, 
                                                      sig_lens)

            sigs_list.extend(sigs)
            segment_offset = range_t[0]
            
            for seg_idx, sig_len in enumerate(sig_lens):
                seg_len_sec = float(sig_len / self.sr)
                start_abs_sec = round(float(segment_offset + seg_idx*self.embed_seg_hop), ROUND)
                end_abs_sec = round(float(segment_offset + seg_idx*self.embed_seg_hop + seg_len_sec), ROUND)
                sig_rangel_list.append([start_abs_sec, end_abs_sec])

        return sigs_list, sig_rangel_list

    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        
        unmerged, diar_labels = self._decode_and_cluster(frame, offset=self.offset)
        
        text, char_ts, end_stamp = self.greedy_merge_with_ts(unmerged, self.frame_start)
        # return self.greedy_merge(unmerged), curr_speaker, unmerged, whole_buffer_text, text, char_ts, end_stamp, diar_labels
        return text, char_ts, end_stamp, diar_labels

    
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



if __name__ == "__main__":
    torch.manual_seed(4)
    
    GT_RTTM_DIR="/disk2/scps/rttm_scps/all_callhome_rttm.scp"
    AUDIO_SCP="/disk2/scps/audio_scps/all_callhome.scp"
    ORACLE_VAD="/disk2/scps/oracle_vad/modified_oracle_callhome_ch109.json"
    reco2num='/disk2/datasets/modified_callhome/RTTMS/reco2num.txt'
    SEG_LENGTH=1.5
    SEG_SHIFT=0.75
    SPK_EMBED_MODEL="/home/taejinp/gdrive/model/ecapa_tdnn/ecapa_tdnn.nemo"
    DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/online_ch109_oracle_vad'
    reco2num=2
    session_name = "en_4092"  ### up to 0.0183

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
    
    cfg_diar = hydra.compose(config_name="speaker_diarization.yaml", overrides=overrides)

    online_diar_model = OnlineClusteringDiarizer(cfg=cfg_diar)
    online_diar_model.prepare_diarization()
    
    
    cfg, asr_model = load_ASR_model()

    SAMPLE_RATE = 16000
    FRAME_LEN = 1.0
    asr = Frame_ASR_DIAR(asr_model, online_diar_model,
                         model_definition = {
                               'sample_rate': SAMPLE_RATE,
                               'AudioToMelSpectrogramPreprocessor': cfg.preprocessor,
                               'JasperEncoder': cfg.encoder,
                               'labels': cfg.decoder.vocabulary
                         },
                         frame_len=FRAME_LEN, frame_overlap=2, 
                         offset=4)

    asr.reset()
    asr.use_offline_asr = True
    asr.diar_buffer_length_sec = 60

    for uniq_key, dcont in online_diar_model.AUDIO_RTTM_MAP.items():
        if uniq_key == session_name:
            samplerate, sdata = wavfile.read(dcont['audio_path'])
            asr.curr_uniq_key = uniq_key
            asr.rttm_file_path = dcont['rttm_path']
            
            if asr.use_offline_asr:
                # Infer log prob at once to maximize the ASR accuracy
                asr.offline_logits = asr.asr_model.transcribe([dcont['audio_path']], logprobs=True)[0]

                # Pad zeros to sync with online buffer with incoming frame
                asr.offline_logits = np.vstack((np.zeros((int(4*asr.frame_len/asr.time_stride), asr.offline_logits.shape[1])), asr.offline_logits))
            
            for i in range(int(np.floor(sdata.shape[0]/asr.n_frame_len))):
                callback_sim(uniq_key, i, sdata, frame_count=None, time_info=None, status=None)

