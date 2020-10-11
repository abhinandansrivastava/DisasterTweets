import transformers

DEVICE = "cuda"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10

BERT_PATH = '../input/bert-base-uncased'
TRAINING_FILE = 'train_preprocess_2.csv'
# TRAINING_FILE = 'IMDB_Dataset_1.csv'
MODEL_PATH = 'model_2.bin'

TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
