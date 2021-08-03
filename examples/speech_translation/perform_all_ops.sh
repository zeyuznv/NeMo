set -e
if [ -z "${workdir}"]; then
  workdir=~/data/iwslt/IWSLT-SLT/eval/en-de/IWSLT.tst2019
fi

printf "Creating IWSLT manifest.."
python create_iwslt_manifest.py -a "${workdir}/wavs" \
  -t "${workdir}/IWSLT.TED.tst2019.en-de.en.xml" \
  -o "${workdir}/manifest.json"

printf "\nSplitting audio files.."
python iwslt_split_audio.py -a "${workdir}/wavs" \
  -s "${workdir}/IWSLT.TED.tst2019.en-de.yaml" \
  -d "${workdir}/split"

printf "\nTranscription.."
workdir="${workdir}" bash transcribe_with_different_models.sh

printf "\nComputing WER.."
workdir="${workdir}" bash compute_all_wers.sh

printf "\nPunctuation and capitalization.."
workdir="${workdir}" bash punc_cap_all.sh

printf "\nTranslation.."
workdir="${workdir}" bash translate.sh

#printf "\nRemoving sound segments.."
#python remove_sound_segments.py -s "${workdir}/IWSLT.TED.tst2019.en-de.en.xml" \
#  -t "${workdir}/IWSLT.TED.tst2019.en-de.de.xml" \
#  -S "${workdir}/IWSLT.TED.tst2019.en-de.en.no_sounds.xml" \
#  -T "${workdir}/IWSLT.TED.tst2019.en-de.de.no_sounds.xml"

printf "\nCreating de ground truth.."
workdir="${workdir}" bash create_de_ground_truth.sh

printf "\nmwerSegmenting.."
workdir="${workdir}" bash mwerSegmenter.sh

printf "\nPreparing mwer segments for BLEU scoring.."
workdir="${workdir}" bash prepare_translations_and_references_for_mwer_scoring.sh

printf "\nScoring translations.."
workdir="${workdir}" bash score_translations.sh

set +e