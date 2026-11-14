def doc_to_text(doc):
    return (
        f"Essay:\n{doc['essay']}\n\n"
        "Find the spelling, grammar and syntax errors in this essay. Answer in JSON, grouping errors by "
        'type; for each error give "error", "correction", "sentence" and "improvement".\n'
        "Feedback:\n"
    )
