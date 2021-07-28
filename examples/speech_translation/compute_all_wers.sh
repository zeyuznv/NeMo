set -e -x
> "${output}"
for f in "${pred_dir}"/*; do
  if [ "${f: -9}" == ".manifest" ]; then
    wer="$(python wer_between_2_manifests.py "${f}" "${ground_truth}")"
    fn="$(basename "${f}")"
    model_name="${fn%.*}"
    echo "${model_name} ${wer}" | tee -a "${output}"
  fi
done
set +e +x
