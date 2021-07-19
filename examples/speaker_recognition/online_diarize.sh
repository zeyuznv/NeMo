export HYDRA_FULL_ERROR=1
# speakernet-M-2N-64bs-200e-0.02lr-0.1wr-Vox1Vox2Fisher
EXP_NAMES=" 
ami_finetune_lr_0.001_ep_20.nemo
"
BASE_PATH=/home/taejinp/projects/NeMo/examples/speaker_recognition

###reco2num='/disk2/datasets/amicorpus_lapel/lapel_files/reco2num_test.txt'

### >>>>>>>>>> AMI START
#AUDIO_SCP="/disk2/datasets/amicorpus_lapel/lapel_files/amicorpus_test_wav.scp"
#ORACLE_VAD="/disk2/datasets/amicorpus_lapel/lapel_files/oracle_amicorpus_lapel_test_manifest.json"
#GT_RTTM_DIR="/disk2/datasets/amicorpus_lapel/lapel_files/amicorpus_test_rttm.scp"
#SEG_LENGTH=3
#SEG_SHIFT=1.5
#SPK_EMBED_MODEL="speakerverification_speakernet"
#DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/small_ami_oracle_vad'
#DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/ami_oracle_vad'
#DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/ami_system_vad'
#VAD_MODEL='vad_telephony_marblenet'
#VAD_THRES=0.7
reco2num=null
##reco2num='/home/taejinp/nemo_buffer/small_ami_oracle_vad/reco2num_test.txt'
##reco2num=4
### >>>>>>>>>> AMI END


#AUDIO_SCP="/home/taejinp/projects/temp/amicorpus_test_wav.scp"
#ORACLE_VAD="/home/taejinp/projects/temp/oracle_amicorpus_lapel_test_manifest.json"
#GT_RTTM_DIR="/home/taejinp/projects/temp/amicorpus_test_rttm.scp"


# ORACLE_VAD="/disk2/datasets/NIST_SRE_2000_LDC2001S97/NIST_SRE_2000_LDC2001S97_16k/modified_oracle_NIST_callhome.json"
# ORACLE_VAD="/disk2/test_norm.json"
#GT_RTTM_DIR="/disk2/scps/rttm_scps/all_callhome_rttm.scp"
#AUDIO_SCP="/disk2/scps/audio_scps/all_callhome.scp"

### >>>>>>>>>> Callhome AES START
GT_RTTM_DIR="/disk2/scps/rttm_scps/callhome_ch109.rttm"
AUDIO_SCP="/disk2/scps/audio_scps/callhome_ch109.scp"
ORACLE_VAD="/disk2/scps/oracle_vad/modified_oracle_callhome_ch109.json"
reco2num='/disk2/datasets/modified_callhome/RTTMS/reco2num.txt'
SEG_LENGTH=1.5
SEG_SHIFT=0.75
SPK_EMBED_MODEL="speakerdiarization_speakernet"
DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/ch109_oracle_vad'
#DIARIZER_OUT_DIR='/home/taejinp/nemo_buffer/ch109_system_vad'
##VAD_MODEL='vad_telephony_marblenet'
##VAD_THRES=0.7
#reco2num=2
### >>>>>>>>>> Callhome AES END

# AUDIO_SCP="/disk2/datasets/AMI_simulated/ami_16_speakers/wav.scp"
# ORACLE_VAD="/disk2/datasets/AMI_simulated/ami_16_speakers/oracle_manifest.json"
# GT_RTTM_DIR="/disk2/datasets/AMI_simulated/ami_16_speakers/rttm.scp"
# reco2num=null

# AUDIO_SCP="/disk2/TTS/tts_wav.scp"
# ORACLE_VAD="/disk2/TTS/oracle.json"
# GT_RTTM_DIR="/disk2/TTS/tts_rttm.scp"
# reco2num=null
CUDA_VISIBLE_DEVICES=0

mkdir -p $DIARIZER_OUT_DIR
#if [ -d "$DIARIZER_OUT_DIR/pred_rttms" ]
#then
    #rm $DIARIZER_OUT_DIR/pred_rttms/*.rttm
    #rm $DIARIZER_OUT_DIR/pred_rttms/system_rttm_total
#fi


#rm -f result
#touch result
EXP_DIR="/disk2/Fei/"
ORACLE_MODEL="/disk2/jagadeesh/vad_checkpoints/marblenet-I-4N-64bs-50e-FisherAMI_310ms.nemo"
# reco2num="/disk2/datasets/NIST_SRE_2000_LDC2001S97/NIST_SRE_2000_LDC2001S97_16k/reco2num"
#for name in $EXP_NAMES;
#do
	#echo $name
	# result=$(
	python $BASE_PATH/streaming_asr_diar.py \
		diarizer.speaker_embeddings.model_path=$SPK_EMBED_MODEL \
		diarizer.path2groundtruth_rttm_files=$GT_RTTM_DIR \
		diarizer.paths2audio_files=$AUDIO_SCP \
		diarizer.out_dir=$DIARIZER_OUT_DIR \
		diarizer.oracle_num_speakers=$reco2num \
		diarizer.speaker_embeddings.oracle_vad_manifest=$ORACLE_VAD \
        diarizer.speaker_embeddings.window_length_in_sec=$SEG_LENGTH \
        diarizer.speaker_embeddings.shift_length_in_sec=$SEG_SHIFT 
        #diarizer.vad.model_path=$VAD_MODEL \
        #diarizer.vad.threshold=$VAD_THRES
	# ) 2> /dev/null || exit 1
	#out=$(echo $result | tr ']' '\n' | grep 'Cumulative' | awk '{print $(NF-2),$(NF)}')
	#echo $name $out >> result
#done

#AUDIO_SCP="/disk2/callhome_eval.scp"
#diarizer.vad.model_path='vad_marblenet' \
#cat result


#cat $(grep -v '^#' $GT_RTTM_DIR) > $DIARIZER_OUT_DIR/gt_rttm_total
#cat $DIARIZER_OUT_DIR/pred_rttms/*.rttm | tr -s " " > $DIARIZER_OUT_DIR/pred_rttms/system_rttm_total
#sed -i 's/-Headset/-Lapel/' $DIARIZER_OUT_DIR/gt_rttm_total
#~/kaldi/tools/sctk/src/md-eval/md-eval.pl -1 -c 0.25  -r $DIARIZER_OUT_DIR/gt_rttm_total -s $DIARIZER_OUT_DIR/pred_rttms/system_rttm_total > $DIARIZER_OUT_DIR/md_eval_out.txt

#MDEVAL_OUT=$DIARIZER_OUT_DIR/md_eval_out.txt
#der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' $MDEVAL_OUT )
#spk_err=$(grep "SPEAKER ERROR TIME =" $MDEVAL_OUT | grep -oP "\K[0-9]+([.][0-9]+)?" - | sed -n 2p)
#miss=$(grep "MISSED SPEECH =" $MDEVAL_OUT | grep -oP "\K[0-9]+([.][0-9]+)?" - | sed -n 2p)
#false_alarm=$(grep "FALARM SPEECH =" $MDEVAL_OUT | grep -oP "\K[0-9]+([.][0-9]+)?" - | sed -n 2p)
#echo md-eval.pl result - DER: "$der", SPK_ERR: "$spk_err", MISS: "$miss", FALSE_ALARM: "$false_alarm"

