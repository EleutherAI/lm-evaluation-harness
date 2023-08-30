
def process_doc(dataset):

    def _process(doc):
        return {
            "id": doc["query_id"],
            "passage": doc["passage"],
            "question": doc["question"],
            "answers": get_answers(doc),
        }
    return dataset.map(_process)


def get_answers(doc):
    def _flatten_validated_answers(validated_answers):
        """Flattens a dict of lists of validated answers.
        {"number": ['1', '8'], ...}
        -> [{"number": ['1'], ...}, {"number": ['8'], ...}]
        """
        valid_answers = []
        for i in range(len(validated_answers["number"])):
            valid_answers.append(
                {
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                }
            )
        return valid_answers

    answers = []
    answers_set = set()
    candidates = [doc["answer"]] + _flatten_validated_answers(
        doc["validated_answers"]
    )
    for candidate in candidates:
        answer = parse_answer(candidate)
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers

def parse_answer(answer):
    # NOTE: Everything is returned as a tuple for uniformity and hashability.
    if answer["number"] != "":
        return (str(answer["number"]),)
    if answer["spans"] != []:
        return tuple(answer["spans"])
    return (
        " ".join(
            [answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]
        ).strip(),
    )