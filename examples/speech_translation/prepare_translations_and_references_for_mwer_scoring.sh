set -e -x
workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019

python xml_2_text_segs_2_lines.py -i "${workdir}/IWSLT.TED.tst2019.en-de.de.xml" \
  -o "${workdir}/iwslt_de_text_by_segs.txt"

translations_after_mwer="${workdir}/mwerSegmented"


for segmentation in $(ls "${translations_after_mwer}"); do
  for inp_length in $(ls "${translations_after_mwer}/${segmentation}"); do
    for tr_model in $(ls "${translations_after_mwer}/${segmentation}/${inp_length}"); do
      for xml_file in $(ls "${translations_after_mwer}/${segmentation}/${inp_length}/${tr_model}"); do
        if [ "${xml_file##*.}" = "xml" ]; then
          asr_model="${xml_file%.*}"
          save_path="${workdir}/${segmentation}_mwer/${inp_length}/${tr_model}"
          mkdir -p "${save_path}"
          python xml_2_text_segs_2_lines.py \
            -i "${translations_after_mwer}/${segmentation}/${inp_length}/${tr_model}/${xml_file}" \
            -o "${save_path}/${asr_model}.txt"
        fi
      done
    done
  done
done
set +e +x