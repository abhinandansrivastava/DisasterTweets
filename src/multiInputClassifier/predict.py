import config
import dataset
from model import BERTBaseUncased
import transformers
import torch.nn as nn
import torch


DEVICE = config.DEVICE
MODEL = BERTBaseUncased()
MODEL.load_state_dict(torch.load(config.MODEL_PATH))


def predict(sentence,keyword,MODEL):
    MODEL.to(DEVICE)
    MODEL.eval()
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        keyword,
        add_special_token = True,
        max_len = max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()

    return outputs[0][0]


# sentence = 'Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all'
# keyword = 'nolocation'
#
# print(predict(sentence,keyword,MODEL))