set -e -x


work_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
reference="${work_dir}/iwslt_de_text.txt"
translated_dirs=( translated_transcripts_segmented translated_transcripts_not_segmented )
outputs=( bleu_scores_segmented.txt bleu_scores_not_segmented.txt )

for i in {0..1}; do
  translated_dir="${work_dir}/${translated_dirs[i]}"
  output="${outputs[i]}"
  > "${output}"
  for d in "${translated_dir}"/*; do
    first_level="$(basename "${d}")"
    echo "${first_level}" | tee -a "${output}"
    for m in "${d}"/*; do
      second_level="$(basename "${m}")"
      echo "    ${second_level}" | tee -a "${output}"
      for mm in "${m}"/*; do
        third_level="$(basename "${mm}")"
        bleu=$(sacrebleu "${reference}" -i "${mm}" -m bleu -b -w 4)
        echo "         ${third_level} ${bleu}" | tee -a "${output}"
      done
    done
  done
done

set +e +x