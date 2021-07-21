work_dir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019

python prepare_iwslt_text_for_translation.py -a "${work_dir}/manifest.json" \
    -t "${work_dir}/IWSLT.TED.tst2019.en-de.de.xml" \
    -o "${work_dir}/iwslt_de_text.txt"
