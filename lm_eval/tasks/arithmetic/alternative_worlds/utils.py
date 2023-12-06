import re

# Original Prompt
# Question: What is (9 + 8) * 2? Answer:
def style_00(docs):

    # What is (9 + 8) * 2?
    return docs["context"]


def style_01(docs):

    # What is (9 + 8) * 2?
    return docs["context"].replace("Question: ", "").replace(" Answer:", "")


def style_02(docs):

    # Q: What is (9 + 8) * 2? A:
    return docs["context"].replace("Question: ", "Q: ").replace(" Answer:", " A:")


def style_03(docs):

    # Solve (9 + 8) * 2.
    return (
        docs["context"].replace("Question: What is", "Solve").replace(" Answer:", ".")
    )


def style_04(docs):

    # (9 + 8) * 2 =
    return docs["context"].replace("Question: What is ", "").replace(" Answer:", " =")


def style_05(docs):

    # What is (9 + 8) * 2? Answer:
    return docs["context"].replace("Question: ", "")
