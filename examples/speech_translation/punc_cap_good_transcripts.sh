good_transcript_models=(
  stt_en_citrinet_1024_gamma_0_25
  stt_en_citrinet_1024
  stt_en_citrinet_256_gamma_0_25
  stt_en_citrinet_256
  stt_en_citrinet_512_gamma_0_25
  stt_en_citrinet_512
  stt_en_jasper10x5dr
)

work_dir=~/data/iwslt/IWSLT-SLT/eval/en-de

for m in "${good_transcript_models[@]}"; do
  python punc_cap.py -a "${work_dir}/manifest.json" \
    -p "${work_dir}/transcripts/${m}.manifest" \
    -o "${work_dir}/punc_transripts/${m}.txt"
done