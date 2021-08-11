set -e
if [ -z "${workdir}" ]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi
output_dir="${workdir}/BLEU_by_docs"
scores="${output_dir}/scores.txt"
references="${output_dir}/references"
doc_translations="${output_dir}/doc_translations"

python xml_2_separate_files.py -i "${workdir}/IWSLT.TED.tst2019.en-de.de.xml" \
  -o "${references}"

for segmentation in $(ls "${translations_after_mwer}"); do
  echo "${segmentation}" >> "${scores}"
  for inp_length in $(ls "${translations_after_mwer}/${segmentation}"); do
    echo "    ${inp_length}" >> "${scores}"
    for tr_model in $(ls "${translations_after_mwer}/${segmentation}/${inp_length}"); do
      echo "        ${tr_model}" >> "${scores}"
      for xml_file in $(ls "${translations_after_mwer}/${segmentation}/${inp_length}/${tr_model}"); do
        if [ "${xml_file##*.}" = "xml" ]; then
          asr_model="${xml_file%.*}"
          echo "            ${asr_model}" >> "${scores}"
          python xml_2_separate_files.py \
            -i "${translations_after_mwer}/${segmentation}/${inp_length}/${tr_model}/${xml_file}" \
            -o "${doc_translations}/${segmentation}/${inp_length}/${tr_model}/${asr_model}"
          for doc_file in $(ls "${doc_translations}/${segmentation}/${inp_length}/${tr_model}/${asr_model}"); do
            bleu=$(sacrebleu "${references}/${doc_file}" \
              -i "${doc_translations}/${segmentation}/${inp_length}/${tr_model}/${asr_model}/${doc_file}" \
              -m bleu \
              -b \
              -w 4)
            echo "                ${doc_file%.*} $(bleu)" >> "${scores}"
          done
        fi
      done
    done
  done
done
set +e