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

for m in "${good_transcript_models[@]}"; do
  python translate_iwslt.py -p "~/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo" \
    -i "${work_dir}/punc_transcripts/${m}.manifest" \
    -o "${work_dir}/translated_transcripts/${m}.txt"
done