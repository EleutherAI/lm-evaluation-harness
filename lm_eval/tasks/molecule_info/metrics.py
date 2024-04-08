import re
def rounded_acc(predictions, references):
    """
    filter "The answer is", extract the number, round it and compare to the reference
    """
    print(predictions, references)
    try:
        # extract "The answer is (\\-?[0-9\\.\\,]*[0-9]+\\.?)" from the prediction
        predictions = predictions[0]
        predictions = re.findall(r'The answer is (\-?[0-9\.]*[0-9]+\.?)', predictions)
        # if no match, return 0
        if not predictions:
            print('no match')
            return 0
        # keep the last match
        predictions = predictions[-1]
        number = predictions
        if number[-1] == '.':
            number = number[:-1]
        prediction = float(number)
    except Exception:
        return 0
    reference = float(''.join(references[0]))
    print(prediction, reference)
    if int(prediction+0.5) == int(reference+0.5):
        return 1
    if abs(prediction-reference)<1:
        return 0.5
    return 0
    # return (predictions[0], references[0])


def acc(predictions, references):
    print(predictions, references)
    try:
        prediction = float(predictions[0])
    except Exception:
        return 0
    reference = float(references[0])
    print(prediction, reference)
    return abs(prediction-reference) < 0.01  # 1% tolerance
