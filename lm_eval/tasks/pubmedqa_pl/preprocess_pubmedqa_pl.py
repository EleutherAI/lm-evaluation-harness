def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    return "Kontekst: {}\nPytanie: {}\nOdpowiedź:".format(
        ctxs,
        doc["QUESTION"],
    )
