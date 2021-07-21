#!/usr/bin/bash
gpu_models=(
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

cpu_models=(
  stt_en_conformer_ctc_small
  stt_en_conformer_ctc_medium
  stt_en_conformer_ctc_large
  stt_en_conformer_ctc_small_ls
  stt_en_conformer_ctc_medium_ls
  stt_en_conformer_ctc_large_ls
)

echo "Creating output directory ${output_dir}"
mkdir -p "${output_dir}"
for model_checkpoint in "${gpu_models[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${model_checkpoint}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_dir}/${model_checkpoint}.manifest" \
    cuda=true \
    batch_size=1
done
for model_checkpoint in "${cpu_models[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${model_checkpoint}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_dir}/${model_checkpoint}.manifest" \
    cuda=false \
    batch_size=1
done
