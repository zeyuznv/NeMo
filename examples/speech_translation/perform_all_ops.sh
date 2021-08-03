set -e
if [ -z "${workdir}"]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi

printf "Creating IWSLT manifest.."
python create_iwslt_manifest.py -a "${workdir}/wavs" \
  -t "${workdir}/IWSLT.TED.tst2019.en-de.en.xml" \
  -o "${workdir}/manifest.json"

printf "\n\nSplitting audio files..\n"
python iwslt_split_audio.py -a "${workdir}/wavs" \
  -s "${workdir}/IWSLT.TED.tst2019.en-de.yaml" \
  -d "${workdir}/split"

printf "\n\nTranscription..\n"
workdir="${workdir}" bash transcribe_with_different_models.sh

printf "\n\nComputing WER..\n"
workdir="${workdir}" bash compute_all_wers.sh

printf "\n\nPunctuation and capitalization..\n"
workdir="${workdir}" bash punc_cap_all.sh

printf "\n\nTranslation..\n"
workdir="${workdir}" bash translate.sh

#printf "\nRemoving sound segments.."
#python remove_sound_segments.py -s "${workdir}/IWSLT.TED.tst2019.en-de.en.xml" \
#  -t "${workdir}/IWSLT.TED.tst2019.en-de.de.xml" \
#  -S "${workdir}/IWSLT.TED.tst2019.en-de.en.no_sounds.xml" \
#  -T "${workdir}/IWSLT.TED.tst2019.en-de.de.no_sounds.xml"

printf "\n\nCreating de ground truth..\n"
workdir="${workdir}" bash create_de_ground_truth.sh

printf "\n\nmwerSegmenting..\n"
workdir="${workdir}" bash mwerSegmenter.sh

printf "\n\nPreparing mwer segments for BLEU scoring..\n"
workdir="${workdir}" bash prepare_translations_and_references_for_mwer_scoring.sh

printf "\n\nScoring translations..\n"
workdir="${workdir}" bash score_translations.sh

set +e