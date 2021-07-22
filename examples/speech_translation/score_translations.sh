good_transcript_models=(
  stt_en_citrinet_1024_gamma_0_25
  stt_en_citrinet_1024
  stt_en_citrinet_256_gamma_0_25
  stt_en_citrinet_256
  stt_en_citrinet_512_gamma_0_25
  stt_en_citrinet_512
  stt_en_jasper10x5dr
)

work_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
reference=iwslt_de_text.txt
translated_dir=translated_transcripts
output=bleu_scores.txt

for m in "${good_transcript_models[@]}"; do
  bleu=$(sacrebleu "${work_dir}/${reference}" -i "${work_dir}/${translated_dir}/${m}.txt" -m bleu -b -w 4)
  echo "${m} ${bleu}" | tee -a "${output}"
done