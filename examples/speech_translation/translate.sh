set -e

transcript_models_not_only_segmented_data=(
  QuartzNet15x5Base-En
  stt_en_jasper10x5dr
  stt_en_citrinet_256
  stt_en_citrinet_512
  stt_en_citrinet_1024
  stt_en_citrinet_256_gamma_0_25
  stt_en_citrinet_512_gamma_0_25
  stt_en_citrinet_1024_gamma_0_25
  CitriNet-1024-8x-Stride-Gamma-0.25.nemo
  Citrinet-1024-SPE-Unigram-1024-Jarvis-ASRSet-2.0_no_weight_decay_e250-averaged.nemo
)

transcript_models_only_segmented_data=(
  stt_en_conformer_ctc_small
  stt_en_conformer_ctc_medium
  stt_en_conformer_ctc_large
  stt_en_conformer_ctc_small_ls
  stt_en_conformer_ctc_medium_ls
  stt_en_conformer_ctc_large_ls
  sel_jarvisasrset_d512_adamwlr2_wd0_aug10x0.05_sp128_500e-last.nemo
)

if [ -z "${workdir}" ]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi
translated_dirs=( translated_transcripts_segmented translated_transcripts_not_segmented )

punc_transcripts=( "${workdir}/punc_transcripts_segmented_input" "${workdir}/punc_transcripts_not_segmented_input" )
translation_checkpoints=( ~/checkpoints/wmt21_en_de_backtranslated_24x6_averaged.nemo )
#translation_ngc_models=( nmt_en_de_transformer12x2 )
declare -a translation_ngc_models

for i in {0..1}; do
  if [ "${i}" -eq 0 ]; then
    good_transcript_models=("${transcript_models_not_only_segmented_data[@]}" "${transcript_models_only_segmented_data[@]}")
  else
    good_transcript_models=("${transcript_models_not_only_segmented_data[@]}")
  fi
  translated_dir="${translated_dirs[i]}"
  punc_transcripts_dir="${punc_transcripts[i]}"
  long_segments_result="${workdir}/${translated_dir}/long_segments"
  one_sentence_segments_result="${workdir}/${translated_dir}/one_sentence_segments"
  mkdir -p "${long_segments_result}"
  mkdir -p "${one_sentence_segments_result}"
  for ckpt in "${translation_checkpoints[@]}"; do
    model_name=$(basename "${ckpt}")
    mkdir -p "${long_segments_result}/${model_name}"
    for m in "${good_transcript_models[@]}"; do
      python translate_iwslt.py -p "${ckpt}" \
        -i "${punc_transcripts_dir}/${m}.txt" \
        -o "${long_segments_result}/${model_name}/${m}.txt"
    done

    mkdir -p "${one_sentence_segments_result}/${model_name}"
    for m in "${good_transcript_models[@]}"; do
      python translate_iwslt.py -p "${ckpt}" \
        -i "${punc_transcripts_dir}/${m}.txt" \
        -o "${one_sentence_segments_result}/${model_name}/${m}.txt" \
        -s
    done
  done

  if [ "${#translation_ngc_models[@]}" -ne 0 ]; then
    for ngc_model in "${translation_ngc_models[@]}"; do
      mkdir -p "${long_segments_result}/${ngc_model}"
      for m in "${good_transcript_models[@]}"; do
        python translate_iwslt.py -m "${ngc_model}" \
          -i "${punc_transcripts_dir}/${m}.txt" \
          -o "${long_segments_result}/${ngc_model}/${m}.txt"
      done

      mkdir -p "${one_sentence_segments_result}/${ngc_model}"
      for m in "${good_transcript_models[@]}"; do
        python translate_iwslt.py -m "${ngc_model}" \
          -i "${punc_transcripts_dir}/${m}.txt" \
          -o "${one_sentence_segments_result}/${ngc_model}/${m}.txt" \
          -s
      done
    done
  fi
done

set +e