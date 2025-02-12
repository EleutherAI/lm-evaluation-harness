import torch
from sklearn.metrics import f1_score, precision_score, recall_score


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
    # get |A| and |T| to compute the final precision and recall using a lambda function
    A = sum([p[1] for p in predictions])
    T = sum([p[2] for p in predictions])
    # compute the final precision and recall
    if A == 0:
        prec = sum([p[0] for p in predictions]) / 1
    else:
        prec = sum([p[0] for p in predictions]) / A
    if T == 0:
        rec = sum([p[0] for p in predictions]) / 1
    else:
        rec = sum([p[0] for p in predictions]) / T
    # compute the final F1 score
    f1 = 0
    if prec + rec != 0:
        f1 = (2 * prec * rec) / (prec + rec)
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


# ------------------------ DOCUMENT DATING ---------------------------


def _aggreg_dd(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore
