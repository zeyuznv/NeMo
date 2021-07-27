#!/usr/bin/bash
models_working_on_not_split_data=(
  QuartzNet15x5Base-En
  stt_en_jasper10x5dr
  stt_en_citrinet_256
  stt_en_citrinet_512
  stt_en_citrinet_1024
  stt_en_citrinet_256_gamma_0_25
  stt_en_citrinet_512_gamma_0_25
  stt_en_citrinet_1024_gamma_0_25
)
#  stt_en_conformer_ctc_small
#  stt_en_conformer_ctc_medium
#  stt_en_conformer_ctc_large
#  stt_en_conformer_ctc_small_ls
#  stt_en_conformer_ctc_medium_ls
#  stt_en_conformer_ctc_large_ls
#)

models_working_on_split_data=(
  stt_en_conformer_ctc_small
  stt_en_conformer_ctc_medium
  stt_en_conformer_ctc_large
  stt_en_conformer_ctc_small_ls
  stt_en_conformer_ctc_medium_ls
  stt_en_conformer_ctc_large_ls
)


nemo_asr_checkpoints=(
  CitriNet-1024-8x-Stride-Gamma-0.25.nemo
  Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-2.0_no_weight_decay_e250-averaged.nemo
  sel_jarvisasrset_d512_adamwlr2_wd0_aug10x0.05_sp128_500e-last.nemo
)

checkpoint_dir=~/checkpoints
echo "Creating output directory ${output_dir}"
mkdir -p "${output_dir}"

for model_checkpoint in "${nemo_asr_checkpoints[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py model_path="${checkpoint_dir}/${model_checkpoint}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_dir}/$(basename "${model_checkpoint}").manifest" \
    cuda=true \
    batch_size=1
done

split_data_path="${audio_dir}/../split"
split_transcripts="${audio_dir}/../split_transcripts"
for model_checkpoint in "${nemo_asr_checkpoints[@]}"; do
  mkdir -p "${split_transcripts}/${model_checkpoint}"
  for f in "${split_data_path}"/*; do
    talk_id=$(basename "${f}")
    if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
      python ~/NeMo/examples/asr/transcribe_speech.py model_path="${checkpoint_dir}/${model_checkpoint}" \
        audio_dir="${f}" \
        output_filename="${split_transcripts}/$(basename "${model_checkpoint}")/${talk_id}.manifest" \
        cuda=true \
        batch_size=4
    fi
  done
done
for pretrained_name in "${models_working_on_split_data[@]}" "${models_working_on_not_split_data[@]}"; do
  mkdir -p "${split_transcripts}/${pretrained_name}"
  for f in "${split_data_path}"/*; do
    talk_id=$(basename "${f}")
    if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
      python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${pretrained_name}" \
        audio_dir="${f}" \
        output_filename="${split_transcripts}/${pretrained_name}/${talk_id}.manifest" \
        cuda=true \
        batch_size=4
    fi
  done
done
python join_split_wav_manifests.py -s "${split_transcripts}" -o "${output_dir}" -n "${audio_dir}"

for pretrained_name in "${models_working_on_not_split_data[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${pretrained_name}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_dir}/${pretrained_name}.manifest" \
    cuda=true \
    batch_size=1
done





#split_data_path="${audio_dir}/../split"
#split_transcripts="${audio_dir}/../split_transcripts"
#for f in "${split_data_path}"/*; do
#  talk_id=$(basename "${f}")
#  if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
#    mkdir -p "${split_transcripts}/${talk_id}"
#    for pretrained_name in "${models_working_on_split_data[@]}"; do
#      python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${pretrained_name}" \
#        audio_dir="${f}" \
#        output_filename="${split_transcripts}/${talk_id}/${pretrained_name}.manifest" \
#        cuda=true \
#        batch_size=4
#    done
#  fi
#done


