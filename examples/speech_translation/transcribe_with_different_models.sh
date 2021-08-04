#!/usr/bin/bash
set -e

pretrained_ngc_not_only_segmented_data=(
  QuartzNet15x5Base-En
  stt_en_jasper10x5dr
  stt_en_citrinet_256
  stt_en_citrinet_512
  stt_en_citrinet_1024
  stt_en_citrinet_256_gamma_0_25
  stt_en_citrinet_512_gamma_0_25
  stt_en_citrinet_1024_gamma_0_25
)

pretrained_ngc_only_segmented_data=(
  stt_en_conformer_ctc_small
  stt_en_conformer_ctc_medium
  stt_en_conformer_ctc_large
  stt_en_conformer_ctc_small_ls
  stt_en_conformer_ctc_medium_ls
  stt_en_conformer_ctc_large_ls
)


nemo_asr_checkpoints_not_only_segmented_data=(
  CitriNet-1024-8x-Stride-Gamma-0.25.nemo
  Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-2.0_no_weight_decay_e250-averaged.nemo
)

nemo_asr_checkpoints_only_segmented_data=(
  sel_jarvisasrset_d512_adamwlr2_wd0_aug10x0.05_sp128_500e-last.nemo
)

checkpoint_dir=~/checkpoints
if [ -z "${workdir}"]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi
output_segmented_no_numbers="${workdir}/transcripts_segmented_input_no_numbers"
output_not_segmented_no_numbers="${workdir}/transcripts_not_segmented_input_no_numbers"
output_segmented="${workdir}/transcripts_segmented_input"
output_not_segmented="${workdir}/transcripts_not_segmented_input"
audio_dir="${workdir}/wavs"
echo "Creating output directories '${output_segmented}', '${output_not_segmented}', '${output_segmented_no_numbers}', '${output_not_segmented_no_numbers}'"
mkdir -p "${output_segmented}" "${output_not_segmented}" "${output_segmented_no_numbers}" "${output_not_segmented_no_numbers}"

for model_checkpoint in "${nemo_asr_checkpoints_not_only_segmented_data[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py model_path="${checkpoint_dir}/${model_checkpoint}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_not_segmented_no_numbers}/$(basename "${model_checkpoint}").manifest" \
    cuda=true \
    batch_size=1
done

split_data_path="${audio_dir}/../split"
split_transcripts="${audio_dir}/../split_transcripts"
for model_checkpoint in "${nemo_asr_checkpoints_not_only_segmented_data[@]}" "${nemo_asr_checkpoints_only_segmented_data}"; do
  mkdir -p "${split_transcripts}/$(basename "${model_checkpoint}")"
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
for pretrained_name in "${pretrained_ngc_only_segmented_data[@]}" "${pretrained_ngc_not_only_segmented_data[@]}"; do
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
python join_split_wav_manifests.py -s "${split_transcripts}" -o "${output_segmented_no_numbers}" -n "${audio_dir}"

for pretrained_name in "${pretrained_ngc_not_only_segmented_data[@]}"; do
  python ~/NeMo/examples/asr/transcribe_speech.py pretrained_name="${pretrained_name}" \
    audio_dir="${audio_dir}" \
    output_filename="${output_not_segmented_no_numbers}/${pretrained_name}.manifest" \
    cuda=true \
    batch_size=1
done

for inp_manifest_dir in "${output_segmented_no_numbers}" "${output_not_segmented_no_numbers}"; do
  if [ "${inp_manifest_dir}" = "${output_segmented_no_numbers}" ]; then
    out_manifest_dir="${output_segmented}"
  else
    out_manifest_dir="${output_not_segmented}"
  fi
  for inp_manifest_path in "${inp_manifest_dir}"/*; do
    python text_to_numbers.py -i "${inp_manifest_path}" -o "${out_manifest_dir}/$(basename "${inp_manifest_path}")"
  done
done

set +e
