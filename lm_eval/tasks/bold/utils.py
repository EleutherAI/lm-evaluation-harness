import numpy

def doc_to_text(doc):
    # inputs = doc["prompts"][0]
    inputs = numpy.random.choice(doc["prompts"])
    # inputs = numpy.random.choice(doc["prompts"])
    # inputs = " ".join(doc["prompts"]).replace("\n", " ")
    # inputs = " ".join(inputs.strip().split())

    return inputs


def doc_to_target(doc):
    # targets = doc["wikipedia"][0]
    targets = numpy.random.choice(doc["wikipedia"])
    # targets = " ".join(doc["wikipedia"]).replace("\n", "")
    # targets = " ".join(targets.strip().split())

    return targets