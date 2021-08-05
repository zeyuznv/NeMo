set -e
if [ -z "${workdir}" ]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi

python prepare_iwslt_text_for_translation.py -a "${workdir}/manifest.json" \
    -t "${workdir}/IWSLT.TED.tst2019.en-de.de.xml" \
    -o "${workdir}/iwslt_de_text.txt" \
    -j

set +e
