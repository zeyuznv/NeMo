good_transcript_models=(
  stt_en_citrinet_1024_gamma_0_25
  stt_en_citrinet_1024
)
#  stt_en_citrinet_256_gamma_0_25
#  stt_en_citrinet_256
#  stt_en_citrinet_512_gamma_0_25
#  stt_en_citrinet_512
#  stt_en_jasper10x5dr
#)

work_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
reference="${work_dir}/iwslt_de_text.txt"
translated_dir="${work_dir}/translated_transcripts"
output=bleu_scores.txt

> "${output}"

for d in "${translated_dir}"/*; do
  first_level="$(basename "${d}")"
  echo "${first_level}" | tee -a "${output}"
  for m in "${d}"/*; do
    second_level="$(basename "${m}")"
    bleu=$(sacrebleu "${reference}" -i "${m}.txt" -m bleu -b -w 4)
    echo "    ${second_level} ${bleu}" | tee -a "${output}"
  done
done