from textdistance import levenshtein
from transformers import AutoTokenizer

# Change this tokenizer to fit with the model you are using.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b", max_new_tokens=128)


def token_edit_distance(references, predictions, **kwargs):
    print(references)
    print(predictions)
    print("###")
    ref_tokens = tokenizer.encode(references[0])
    pred_tokens = tokenizer.encode(predictions[0])
    return levenshtein.distance(ref_tokens, pred_tokens)
