
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re


def post_process(doc, results):
    gold = doc["label"]
    label = results[0].strip()
    return {"eval": (label, gold)}

#
# repo: https://pyzone.dev/word-error-rate-in-python
#
def wer(ref, hyp, debug=True):
    r = ref
    h = hyp
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = (
                    costs[i - 1][j - 1] + SUB_PENALTY
                )  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round((numSub + numDel + numIns) / (float)(len(r)), 3)
    if debug:
        return {
            "WER": wer_result,
            "numCor": numCor,
            "numSub": numSub,
            "numIns": numIns,
            "numDel": numDel,
            "numCount": len(r),
        }
    else:
        return {"WER": wer_result}

def evaluate(items):
    predicted_labels, true_labels = zip(*items)
    # Flatten sentences into a long list of words
    hyp = []
    ref = []
    for t, p in zip(true_labels, predicted_labels):
        if p is None:
            # Use undiacritized word in case of prediction failiure
            p = re.sub(r"[ًٌٍَُِّْ]", "", t).split()
        else:
            p = p.split()

        t = t.split()

        # If prediction is missing tokens, pad with empty tokens
        if len(p) < len(t):
            for i in range(len(p) - len(t)):
                hyp.append("")

        # If prediction has extra tokens, only consider the first
        # N tokens, where N == number of gold tokens
        hyp += p[: len(t)]
        ref += t
    return wer(ref, hyp, False)