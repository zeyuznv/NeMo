set -e

pretrained_ngc_not_only_segmented_data=(
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

pretrained_ngc_only_segmented_data=(
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

segmented_transcripts_dir="${workdir}/transcripts_segmented_input"
unsegmented_transcripts_dir="${workdir}/transcripts_not_segmented_input"

punc_segmented_transcripts_dir="${workdir}/punc_transcripts_segmented_input"
punc_unsegmented_transcripts_dir="${workdir}/punc_transcripts_not_segmented_input"

for m in "${pretrained_ngc_not_only_segmented_data[@]}" "${pretrained_ngc_only_segmented_data[@]}"; do
  python punc_cap.py -a "${workdir}/manifest.json" \
    -p "${segmented_transcripts_dir}/${m}.manifest" \
    -o "${punc_segmented_transcripts_dir}/${m}.txt"
done
for m in "${pretrained_ngc_not_only_segmented_data[@]}"; do
  python punc_cap.py -a "${workdir}/manifest.json" \
    -p "${unsegmented_transcripts_dir}/${m}.manifest" \
    -o "${punc_unsegmented_transcripts_dir}/${m}.txt"
done

set +e