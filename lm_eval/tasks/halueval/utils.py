# Main reference: https://github.com/RUCAIBox/HaluEval/blob/main/evaluation/evaluate.py

QA_INSTURCTIONS = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the answer misunderstands the question context and intention.
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Answer#: American Hairless Terrier
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#: Yes
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No
#Question#: What profession do Kōbō Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""

DIALOGUE_INSTRUCTIONS = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the true entity in the response is replaced with a highly similar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""

SUMMARIZATION_INSTRUCTIONS = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.
#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an "extremely sad example of an abandoned and neglected exotic pet". Inspector Selina Chan said: "It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. "We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. "Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006." She added it was illegal to release non-native species into the wild.
#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.
#Your Judgement#: Yes

You are trying to determine if there exists some non-factual and incorrect information in the summary.  
#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". "Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian," he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. "Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose," he said on Tuesday.
#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the summary and the document.
#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: "I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. "But let's not forget that there's two other families who are still waiting for their loved ones to be returned." The building was due for demolition when it partially collapsed in February.
#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.
#Your Judgement#: Yes

You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\""."""


def doc_to_text_qa(doc: dict[str, str]) -> str:
    # prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
    doc_text = QA_INSTURCTIONS + "\n\n#Knowledge: " + doc["knowledge"] + "\n#Question#: " + doc["question"] + "\n#Answer#: " + doc["answer"] + "\n#Your Judgement#:"
    return doc_text


def doc_to_text_dialogue(doc: dict[str, str]) -> str:
    # prompt = instruction + "\n\n#Dialogue History#: " + dialog + "\n#Response#: " + response + "\n#Your Judgement#:"
    doc_text = DIALOGUE_INSTRUCTIONS + "\n\n#Knowledge: " + doc["knowledge"] + "\n#Dialogue History#: " + doc["dialogue_history"] + "\n#Response#: " + doc["response"] + "\n#Your Judgement#:"
    return doc_text


def doc_to_text_summarization(doc: dict[str, str]) -> str:
    # prompt1 = instruction + "\n\n#Document#: " + document
    # prompt2 = "\n#Summary#: " + summary + "\n#Your Judgement#:"
    doc_text_1 = SUMMARIZATION_INSTRUCTIONS + "\n\n#Document#: " + doc["document"]
    doc_text_2 = "\n#Summary#: " + doc["summary"] + "\n#Your Judgement#:"
    doc_text = doc_text_1 + doc_text_2
    return doc_text


def doc_to_target_qa(doc: dict[str, str]) -> str:
    return doc['hallucination']


def compute_metrics_qa(gold_answer: str, prediction: str) -> dict[str, float]:
    is_correct = True

    if ("Yes" in prediction and "No" in prediction) or ("Yes" not in prediction and "No" not in prediction):
        is_correct = False
    elif "Yes" in prediction:
        prediction = "yes"
    elif "No" in prediction:
        prediction = "no"

    is_exact = gold_answer == prediction

    res = {"correctness": 1.0 if is_correct else 0.0}
    if is_correct:
        res["em"] = 1.0 if is_exact else 0.0

    return res


def process_results_qa(doc: dict[str, str], results: list[str]):
    # results is e.g., ['Yes']
    gold_list = doc_to_target_qa(doc)
    # gold_list is e.g., 'yes'
    prediction = results[0].strip().split("\n")[0]
    scores = compute_metrics_qa(gold_list, prediction)
    return scores
