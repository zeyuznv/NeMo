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

echo "Creating output directory ${output_dir}"
mkdir -p "${output_dir}"

split_data_path="${audio_dir}/../split"
split_transcripts="${audio_dir}/../split_transcripts"
for model_checkpoint in "${models_working_on_split_data[@]}"; do
  mkdir -p "${split_transcripts}/${model_checkpoint}"
  for f in "${split_data_path}"/*; do
    talk_id=$(basename "${f}")
    if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
      python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${model_checkpoint}" \
        audio_dir="${f}" \
        output_filename="${split_transcripts}/${model_checkpoint}/${talk_id}.manifest" \
        cuda=true \
        batch_size=4
    fi
  done
  python join_split_wav_manifests.py -s "${split_transcripts}/${model_checkpoint}" -o "${output_dir}" -n "${audio_dir}"
done

for model_checkpoint in "${models_working_on_not_split_data[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${model_checkpoint}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_dir}/${model_checkpoint}.manifest" \
    cuda=true \
    batch_size=1
done


#split_data_path="${audio_dir}/../split"
#split_transcripts="${audio_dir}/../split_transcripts"
#for f in "${split_data_path}"/*; do
#  talk_id=$(basename "${f}")
#  if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
#    mkdir -p "${split_transcripts}/${talk_id}"
#    for model_checkpoint in "${models_working_on_split_data[@]}"; do
#      python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${model_checkpoint}" \
#        audio_dir="${f}" \
#        output_filename="${split_transcripts}/${talk_id}/${model_checkpoint}.manifest" \
#        cuda=true \
#        batch_size=4
#    done
#  fi
#done


