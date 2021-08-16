<< 'MULTILINE-COMMENT'
Before running this script you have to split translation using mwerSegmenter.

Parameters of the script are
  translation: path to XML file created using mwerSegmenter
  reference: path to XML file with references. It is named IWSLT.TED.tst2019.en-de.de.xml
  split_translation_dir: path to a directory where split talk translations are saved
  split_reference: path to a directory where split references are saved
  bleu_file: path to file with BLEU scores for individual talks

Usage example:
source compute_bleu_for_separate_talks_one_model.sh
MULTILINE-COMMENT

set -e

translation="$(realpath "$1")"
reference="$(realpath "$2")"
split_translation_dir="$(realpath "$3")"
split_reference_dir="$(realpath "$4")"
bleu_file="$(realpath "$5")"

python xml_2_separate_files.py -i "${reference}" -o "${split_reference_dir}"
python xml_2_separate_files.py -i "${translation}" -o "${split_translation_dir}"

mkdir -p "$(dirname "${bleu_file}")"
> "${bleu_file}"

for doc_file in $(ls "${split_reference_dir}"); do
  bleu=$(sacrebleu "${split_reference_dir}/${doc_file}" \
    -i "${split_translation_dir}/${doc_file}" \
    -m bleu \
    -b \
    -w 4)
  echo "${doc_file%.*} ${bleu}" >> "${bleu_file}"
done
