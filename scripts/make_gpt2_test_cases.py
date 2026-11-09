import random

import torch
import torch.nn.functional as F
import transformers


random.seed(42)


data = [
    "A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)",
    "The term MLP is used ambiguously, sometimes loosely to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation); see § Terminology",
    'Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.[1]',
    "An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function.",
    "MLP utilizes a supervised learning technique called backpropagation for training.[2][3] Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.[4]",
    "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. ",
    "Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.",
    "A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN)",
    "Hello World",
]


model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
tok = transformers.GPT2Tokenizer.from_pretrained("gpt2")

tgs = []

for dat in data:
    random.seed(dat)
    # print(model(tok.encode(dat, return_tensors="pt"))[0][0])

    toks = tok.encode(dat, return_tensors="pt")
    ind = random.randrange(len(toks[0]) - 1)
    logits = F.log_softmax(model(toks)[0], dim=-1)[:, :-1]  # [batch, seq, vocab]

    res = torch.gather(logits, 2, toks[:, 1:].unsqueeze(-1)).squeeze(-1)[0]

    tgs.append(float(res[ind:].sum()))
    print(
        r'("""'
        + tok.decode(toks[0, : ind + 1])
        + r'""", """'
        + tok.decode(toks[0, ind + 1 :])
        + r'"""), '
    )

print(tgs)
