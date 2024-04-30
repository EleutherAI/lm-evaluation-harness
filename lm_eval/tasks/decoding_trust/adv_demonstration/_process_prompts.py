
def generate_prompt(doc):
    # print(doc)
    string = "Here are some examples for the task:\n"
    for pair in doc["examples"]:
        for index, example in enumerate(pair):
            if index % 2 == 0:
                string += example
            else:
                string += (" Answer: " + example + ".\n")
    tmp = doc["input"].removesuffix(" answer:")
    string += "Here's the question: " + tmp + " Answer: "
    # print(string)
    return string