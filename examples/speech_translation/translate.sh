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
translated_dir=translated_transcripts
long_segments_result="${work_dir}/${translated_dir}/long_segments"
one_sentence_segments_result="${work_dir}/${translated_dir}/one_sentence_segments"
punc_transcripts="${work_dir}/punc_transcripts"
translation_checkpoints=( ~/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo )
translation_ngc_models=( nmt_en_de_transformer12x2 )
mkdir -p "${long_segments_result}"
mkdir -p "${one_sentence_segments_result}"

for ckpt in "${translation_checkpoints[@]}"; do
  model_name=$(basename "${ckpt}")
  save_dir="${long_segments_result}/${model_name}"
  mkdir -p "${save_dir}"
  for m in "${good_transcript_models[@]}"; do
    python translate_iwslt.py -p "${ckpt}" \
      -i "${punc_transcripts}/${m}.txt" \
      -o "${save_dir}/${m}.txt"
  done

  mkdir -p "${one_sentence_segments_result}/${ckpt}"
  for m in "${good_transcript_models[@]}"; do
    python translate_iwslt.py -p "${ckpt}" \
      -i "${punc_transcripts}/${m}.txt" \
      -o "${one_sentence_segments_result}/${ckpt}/${m}.txt" \
      -s
  done
done

for ngc_model in "${translation_ngc_models[@]}"; do
  mkdir -p "${long_segments_result}/${ngc_model}"
  for m in "${good_transcript_models[@]}"; do
    python translate_iwslt.py -m "${ngc_model}" \
      -i "${punc_transcripts}/${m}.txt" \
      -o "${long_segments_result}/${ngc_model}/${m}.txt"
  done

  mkdir -p "${one_sentence_segments_result}/${ngc_model}"
  for m in "${good_transcript_models[@]}"; do
    python translate_iwslt.py -m "${ngc_model}" \
      -i "${punc_transcripts}/${m}.txt" \
      -o "${one_sentence_segments_result}/${ngc_model}/${m}.txt" \
      -s
  done
done