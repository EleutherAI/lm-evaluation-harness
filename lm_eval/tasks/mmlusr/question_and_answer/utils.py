import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _helper(doc):
        # Assuming that the 'answer' field in the dataset now contains numbers 0-3 instead of 'A', 'B', 'C', 'D'
        answer_list = ["A", "B", "C", "D"]
        # Convert numeric index to corresponding letter
        answer_index = int(doc["answer"])  # Make sure the answer is an integer
        answer_letter = answer_list[answer_index]

        out_doc = {
            "questions": doc["question"],
            "choices": [doc["choice1"], doc["choice2"], doc["choice3"], doc["choice4"]],
            "answer": answer_letter,  # Include the letter for clarity
        }
        return out_doc

    return dataset.map(_helper)
