import evaluate


def squad_f1(items):
    unzipped_list = list(zip(*items))
    print(unzipped_list)
    ref_squad, pred_squad = unzipped_list[0], unzipped_list[1]
    reference, prediction = [], []
    for index in range(len(reference)):
        pred_dict = {'prediction_text': str(reference[index]), 'id': str(index)}
        ref_dict = {'answers': {'answer_start': [1], 'text': str(prediction[index])}, 'id': str(index)}
        reference.append(pred_dict)
        prediction.append(ref_dict)

    squad_metric = evaluate.load("squad")
    results_squad = squad_metric.compute(predictions=pred_squad, references=ref_squad)
    return round(results_squad['f1'], 2)
