
hypotheses_folder="/home/taejinp/gdrive/audio_data/AMI/AMI_ASR_mixed_hypotheses_quartz"
 dataset_name="AMI_mixed"
input_json_path=/home/taejinp/Downloads/ASR_exp/AMI_split/AMI_mixed_test_manifest.json

#hypotheses_folder="/home/taejinp/gdrive/audio_data/CHAES_979711_T14/transcrpt/mixed_hypotheses_quartz/"
#dataset_name="CH109_mixed"
#input_json_path=/home/taejinp/Downloads/ASR_exp/AMI_split/CH109_mixed_test_manifest.json

#hypotheses_folder="/home/taejinp/gdrive/audio_data/AMI/AMI_ASR_indiv_ch_hypotheses_quartz"
#dataset_name="AMI_split"
#input_json_path=/home/taejinp/Downloads/ASR_exp/AMI_split/AMI_split_test_manifest.json

#hypotheses_folder="/home/taejinp/gdrive/audio_data/CHAES_979711_T14/transcrpt/split_hypotheses_quartz"
#dataset_name="CH109_split"
#input_json_path="/home/taejinp/Downloads/ASR_exp/CH109_split/CH109_split_test_manifest.json"

sctk_dir=/home/taejinp/projects/SCTK
input_json_path="/home/taejinp/Downloads/ASR_exp/"$dataset_name"/"$dataset_name"_test_manifest.json"
python speech_to_text_sclite.py --dataset $input_json_path \
								--sctk_dir $sctk_dir \
								--out_dir $hypotheses_folder \
								--glm en20000405_hub5.glm 
								
