set -e


if [ -z "${workdir}" ]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi
references=(
  iwslt_de_text.txt
  iwslt_de_text.txt
  iwslt_de_text_by_segs.txt
  iwslt_de_text_by_segs.txt
)
translated_dirs=(
  translated_transcripts_segmented
  translated_transcripts_not_segmented
  translated_transcripts_segmented_mwer
  translated_transcripts_not_segmented_mwer
)
outputs=( bleu_scores_segmented.txt bleu_scores_not_segmented.txt bleu_scores_segmented_mwer.txt bleu_scores_not_segmented_mwer.txt )

for i in {0..3}; do
  translated_dir="${workdir}/${translated_dirs[i]}"
  reference="${workdir}/${references[i]}"
  output="${outputs[i]}"
  if [[ -d "${translated_dir}" && -f "${reference}" ]]; then
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
  fi
done

set +e