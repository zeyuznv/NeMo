set -e

dataset_dir="$1"
asr_model="$2"  # Path to checkpoint or NGC pretrained name
translation_model="$3"
output_dir="$4"
segmented="$5"  # 1 or 0
mwerSegmenter="$6"  # 1 or 0


audio_dir="${dataset_dir}/wavs"
model_name="$(basename "${asr_model}")"



if [ "${asr_model: -5}" -eq ".nemo" ]; then
  asr_model_argument_name=model_path
else
  asr_model_argument_name=pretrained_name
fi


printf "Creating IWSLT manifest.."
python create_iwslt_manifest.py -a "${audio_dir}" \
  -t "${dataset_dir}/IWSLT.TED.tst2019.en-de.en.xml" \
  -o "${output_dir}/manifest.json"


if [ "${segmented}" -eq 1 ]; then
  printf "\nSplitting audio files.."
  split_data_path="${output_dir}/split"
  python iwslt_split_audio.py -a "${dataset_dir}/wavs" \
    -s "${dataset_dir}/IWSLT.TED.tst2019.en-de.yaml" \
    -d "${split_data_path}"
  fi
  split_transcripts="${dataset_dir}/split_transcripts/${model_name}"
  transcript="${output_dir}/transcripts_segmented_input/${model_name}.manifest"
  mkdir -p "${output_dir}/transcripts_segmented_input"
  for f in "${split_data_path}"/*; do
    talk_id=$(basename "${f}")
    if [[ "${talk_id}" =~ ^[1-9][0-9]*$ ]]; then
      python ~/NeMo/examples/asr/transcribe_speech.py "${asr_model_argument_name}"="${asr_model}" \
        audio_dir="${f}" \
        output_filename="${split_transcripts}/${talk_id}.manifest" \
        cuda=true \
        batch_size=4
    fi
  done
  python join_split_wav_manifests.py -S "${split_transcripts}" -o "${transcript}" -n "${audio_dir}"
else
  if [ "${segmented}" -ne 0 ]; then
    echo "Wrong value '${segmented}' of fifth parameter of 'translate_and_score.sh'. Only '0' and '1' are supported."
    exit 1
  fi
  transcript="${output_dir}/transcripts_not_segmented_input/${model_name}.manifest"
  mkdir -p "${output_dir}/transcripts_not_segmented_input"
  python ~/NeMo/examples/asr/transcribe_speech.py "${asr_model_argument_name}"="${asr_model}" \
    audio_dir="${audio_dir}" \
    output_filename="${transcript}" \
    cuda=true \
    batch_size=1
fi



set +e