import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


def _aggreg_ls(predictions):
    """
    Custom aggregation to compute corpus level metrics for the lexical substitution task
    predictions is a list of tuples (prec, has_answ, has_annotation)
    prec is the precision before dividing by |A|
    has_answ is 0 if the model did not produce any answer
    has_annotation is 0 if the gold answer is empty: no synonims from annotators
    """
    # print("aggreg fn preds:", predictions)
    # get |A| and |T| to compute the final precision and recall using a lambda function
    A = sum([p[1] for p in predictions])
    T = sum([p[2] for p in predictions])
    # compute the final precision and recall
    if A == 0:
        # TODO check if this is correct
        prec = sum([p[0] for p in predictions]) / 1
    else:
        prec = sum([p[0] for p in predictions]) / A
    if T == 0:
        # TODO check if this is correct
        rec = sum([p[0] for p in predictions]) / 1
    else:
        rec = sum([p[0] for p in predictions]) / T
    # compute the final F1 score
    f1 = 0
    if prec + rec != 0:
        f1 = (2 * prec * rec) / (prec + rec)
    # debug purposes

    # print("prec:", [p[0] for p in predictions])
    # print("A:", A)
    # print("T:", T)
    # print("prec final:", prec)
    # print("rec final:", rec)
    # print("f1 final:", f1)
    return f1


def _aggreg_sa_v2(predictions):
    """
    This aggregation considers the sentiment analysis task as a multiple choice one with four classes
    the f1 score is computed as the average of the f1 scores for each class weighted by the number of samples
    See sklearn.metrics.f1_score for more details

    """
    predictions, references = zip(*predictions)
    f1 = f1_score(references, predictions, average="weighted")
    return f1


def _aggreg_sa(predictions):
    """
    Custom aggregation function for the sentiment analysis task
    The original tasks compute the F1 score for each class and then average them
    Since the prompt cast the task to a multple choice one we need to aggregate the results in a different way
    """
    # split the predictions and references in two lists (pred is a tuple)
    predictions, references = zip(*predictions)
    """
    Class 0: positivo -> 'opos': 1, 'oneg': 0
    Class 1: negativo -> 'opos': 0, 'oneg': 1
    etc.
    """

    def _map_to_original_labels(x):
        """
        Return two separate list of labels for opos and oneg
        x is a list of integers
        """
        opos = []
        oneg = []
        for i in x:
            if i == 0:
                # positive
                opos.append(1)
                oneg.append(0)
            elif i == 1:
                # negative
                opos.append(0)
                oneg.append(1)
            elif i == 2:
                # neutral
                opos.append(0)
                oneg.append(0)
            elif i == 3:
                # mixed
                opos.append(1)
                oneg.append(1)
            else:
                pass
        return opos, oneg

    pred_opos, pred_oneg = _map_to_original_labels(predictions)
    ref_opos, ref_oneg = _map_to_original_labels(references)

    opos_f1 = f1_score(ref_opos, pred_opos, average=None)
    opos_f1_c0 = f1_score(ref_opos, pred_opos, average=None)[0]
    if len(opos_f1) > 1:
        opos_f1_c1 = opos_f1[1]
    else:
        opos_f1_c1 = 0

    # print("f1_c0:", opos_f1_c0)
    # print("f1_c1:", opos_f1_c1)

    # oneg class
    oneg_prec_c0, oneg_prec_c1 = precision_score(
        ref_oneg, pred_oneg, labels=[0, 1], average=None
    )
    oneg_rec_c0, oneg_rec_c1 = recall_score(
        ref_oneg, pred_oneg, labels=[0, 1], average=None
    )
    oneg_f1 = f1_score(ref_oneg, pred_oneg, average=None)
    oneg_f1_c0 = f1_score(ref_oneg, pred_oneg, average=None)[0]
    if len(oneg_f1) > 1:
        oneg_f1_c1 = f1_score(ref_oneg, pred_oneg, average=None)[1]
    else:
        oneg_f1_c1 = 0

    """
    opos_f1_c0 = f1_score(ref_opos, pred_opos, average=None)[0]
    # TODO se limit == 1 qui fallisce, inserire try except
    opos_f1_c1 = f1_score(ref_opos, pred_opos, average=None)[1]

    print("f1_c0:", opos_f1_c0)
    print("f1_c1:", opos_f1_c1)
    # oneg class
    oneg_prec_c0, oneg_prec_c1 = precision_score(
        ref_oneg, pred_oneg, labels=[0, 1], average=None
    )
    oneg_rec_c0, oneg_rec_c1 = recall_score(
        ref_oneg, pred_oneg, labels=[0, 1], average=None
    )
    oneg_f1_c0 = f1_score(ref_oneg, pred_oneg, average=None)[0]
    oneg_f1_c1 = f1_score(ref_oneg, pred_oneg, average=None)[1]
    """

    # average f1 score for each class (opos and oneg)
    f1_score_opos = (opos_f1_c0 + opos_f1_c1) / 2
    f1_score_oneg = (oneg_f1_c0 + oneg_f1_c1) / 2
    # average f1 score for the two classes
    f1_final = (f1_score_opos + f1_score_oneg) / 2

    return f1_final


def _aggreg_ner(predictions):
    pred, ref = zip(*predictions)
    # concat all the predictions and references
    all_pred = []
    for p in pred:
        all_pred.extend(p)
    all_ref = []
    for r in ref:
        all_ref.extend(r)
    # compute the F1 score
    f1 = f1_score(all_ref, all_pred, average=None)
    if len(f1) > 1:
        f1_sum = sum(f1[:-1]) / (len(f1) - 1)
    else:
        f1_sum = f1[0]

    return f1_sum


def _aggreg_ht(predictions):
    return headline_score(predictions)


def _aggreg_rel(predictions):
    pred, ref = zip(*predictions)
    # concat all the predictions and references
    all_pred = []
    for p in pred:
        all_pred.extend(p)
    all_ref = []
    for r in ref:
        all_ref.extend(r)
    # compute the F1 score
    f1 = f1_score(all_ref, all_pred, average="macro")
    return f1


HEADLINE_CLASSIFIER_ID = "Hate-speech-CNERG/dehatebert-mono-italian"


class HeadlineScorer:
    def __init__(
        self,
        model_name_or_path: str,
        device="cuda",  # if torch.cuda.is_available() elif ,
        # torch_dtype=torch.bfloat16,
        # use_lora: bool = False,
    ):
        # check if the device is available
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path  # , torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # if use_lora:
        #    self.config = PeftConfig.from_pretrained(model_name_or_path)
        #    self.model = PeftModel.from_pretrained(self.model, model_name_or_path)

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
        for step, batch in tqdm(
            enumerate(loader), desc="Batch", total=len(texts) // batch_size
        ):
            batch.to(self.device)
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions = [self.model.config.id2label[i.item()] for i in predictions]
            final_preds.extend(predictions)

        return final_preds


def headline_score(items):
    references, predictions = list(zip(*items))
    evaluator = HeadlineScorer(HEADLINE_CLASSIFIER_ID)
    print("ht evaluator:", evaluator)
    # preds = evaluator.predict(predictions)
    # is_neutral = [True if p == "neutral" else False for p in preds]
    # score = sum(is_neutral) / len(predictions)
    score = -1
    return score
