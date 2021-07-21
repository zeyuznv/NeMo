for f in "${pred_dir}"/*; do
  if [ "${f: -9}" == ".manifest" ]; then
    wer=$(python wer_between_2_manifests.py "${f}" "${ground_truth}")
    echo "${$(basename "${f}")%.*} ${wer}" >> "${output}"
  fi
done