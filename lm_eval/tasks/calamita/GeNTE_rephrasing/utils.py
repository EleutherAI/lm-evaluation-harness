import torch
from tqdm import tqdm
from datasets import concatenate_datasets, Dataset
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)

GN_CLASSIFIER_ID = "FBK-MT/GeNTE-evaluator"
TOKENIZER_ID = "Musixmatch/umberto-commoncrawl-cased-v1"


class NeutralScorer:
    def __init__(
        self,
        model_name_or_path: str,
        device="cuda",
        torch_dtype=torch.bfloat16,
        use_lora: bool = False,
    ):
        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        if not hasattr(self.tokenizer, "pad_token_id"):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        if use_lora:
            self.config = PeftConfig.from_pretrained(model_name_or_path)
            self.model = PeftModel.from_pretrained(self.model, model_name_or_path)

        self.model.to(device).eval()

    @inference_decorator()
    def predict(self, texts, batch_size=4, num_workers=0):
        data = Dataset.from_dict({"text": texts})
        data = data.map(
            lambda x: self.tokenizer(x["text"], truncation=True),
            batched=True,
            remove_columns=["text"],
        )
        collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

        final_preds = list()
        for _, batch in tqdm(
            enumerate(loader), desc="Batch", total=len(texts) // batch_size
        ):
            batch.to(self.device)
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            id2label = {0: "neutral", 1: "gendered"}
            predictions = [id2label[i.item()] for i in predictions]
            final_preds.extend(predictions)

        return final_preds


def neutrality_score(items):
    references, predictions = list(zip(*items))
    evaluator = NeutralScorer(GN_CLASSIFIER_ID)
    preds = evaluator.predict(predictions)
    is_neutral = [True if p == "neutral" else False for p in preds]
    score = sum(is_neutral) / len(predictions)
    return score


def process_docs(dataset):
    # We assume the GeNTE data files already contain Set-N only examples
    # dataset = dataset.filter(lambda x: x["SET"] == "Set-N")
    return dataset.rename_column("REF-N", "REF_N").rename_column("REF-G", "REF_G")
