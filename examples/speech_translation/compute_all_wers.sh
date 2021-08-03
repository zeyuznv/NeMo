set -e
output_segmented=wer_scores_segmented.txt
output_not_segmented=wer_scores_not_segmented.txt
if [ -z "${workdir}"]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi
pred_dir_segmented="${workdir}/transcripts_segmented_input"
pred_dir_not_segmented="${workdir}/transcripts_not_segmented_input"
wer_by_transcript_and_audio="${workdir}/wer_by_transcript_and_audio"
ground_truth="${workdir}/manifest.json"
> "${output_segmented}"
> "${output_not_segmented}"
for f in "${pred_dir_segmented}"/*; do
  if [ "${f: -9}" == ".manifest" ]; then
    fn="$(basename "${f}")"
    model_name="${fn%.*}"
    wer="$(python wer_between_2_manifests.py "${f}" "${ground_truth}" \
      -o "${wer_by_transcript_and_audio}/segmented/${model_name}.json")"
    echo "${model_name} ${wer}" | tee -a "${output_segmented}"
  fi
done
for f in "${pred_dir_not_segmented}"/*; do
  if [ "${f: -9}" == ".manifest" ]; then
    fn="$(basename "${f}")"
    model_name="${fn%.*}"
    wer="$(python wer_between_2_manifests.py "${f}" "${ground_truth}" \
      -o "${wer_by_transcript_and_audio}/not_segmented/${model_name}.json")"
    echo "${model_name} ${wer}" | tee -a "${output_not_segmented}"
  fi
done
set +e
