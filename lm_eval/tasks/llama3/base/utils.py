import datasets


def process_arc_c_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    COLUMNS = dataset.column_names

    def map_(doc):
        doc["doc_to_text"] = doc["input_final_prompts"][0].strip()[:-2].strip()
        doc["doc_to_choice"] = [
            x.replace("Answer:", "").strip() for x in doc["output_choice_completions"]
        ]
        doc["doc_to_target"] = doc["input_correct_responses"][0].strip()[-1]
        return doc

    return dataset.map(map_, remove_columns=COLUMNS)
