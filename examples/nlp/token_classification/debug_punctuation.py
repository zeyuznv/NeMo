from nemo.collections.nlp.models import PunctuationCapitalizationModel

# to get the list of pre-trained models
PunctuationCapitalizationModel.list_available_models()

# Download and load the pre-trained BERT-based model
model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")

# try the model on a few examples
model.add_punctuation_capitalization(['how are you', 'great how about you'])