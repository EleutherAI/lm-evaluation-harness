from lm_eval.utils import weighted_f1_score


def doc_to_text(doc):
    output = """Read the following text and definitions:

    Text: {tweet}.

    Definitions:
    Hate: Hate speech is public speech that expresses hate or encourages violence towards a person or group based on
    something such as race, religion, gender, ethnicity, sexual orientation or other characteristics.

    Abuse: Abusive and offensive language means verbal messages that use words in an inappropriate way and may include
    but is not limited to swearing, name-calling, or profanity. Offensive language may upset or embarrass people
    because it is rude or insulting

    Normal: Normal language is neither hateful nor abusive or offensive. It does not contain any bad language.

    Which of these definitions (hate, abuse, normal) apply to this tweet?

    """

    text = output.format(tweet=doc["tweet"])
    return text


def macro_f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore


def micro_f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="micro")
    return fscore
