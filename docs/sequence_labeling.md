# Sequence labeling

Sequence labeling datasets come in various formats, but here, we assume a dataset with word-level labels—either in IOB format for tasks like Named Entity Recognition or in word-level tagging for tasks like POS tagging.

If your dataset does not fit this structure, you will need to preprocess it.

# Handling the IOB scheme

## Build a dataset from IOB scheme
We must structure our IOB-labeled dataset in a format that most LLMs can both comprehend and generate. There is a lot of literature showing different prompting strategies, but the most common consist on delimiting chunks with special symbols. For a more exhaustive list of prompting strategies, refer to the [References](#references) section.

Based on existing literature, here we create in-text annotated examples in which each relevant chunk is explicitly marked with `<>` tags. For example: `<person> John Smith </person> plays football .`

These annotated examples serve as our labels, allowing us to later reconstruct the IOB labeling from the provided annotations.

> [!IMPORTANT]  
> Note that we are using tags to annotate within the text. So, if your text contain this kind of tags, this method will not work well. It is highly recommended to remove these characters from your dataset for tagging tasks.

Here is a snippet to illustrate how to prepare an IOB-labeled dataset to be processed by lm-evaluation-harness.

```python
# A few training examples from your dataset to elicit the format we expect
train_texts = [
    "Moncada is a city near Valencia in Spain",
    "The football player Miguel Tendillo Valencia comes in July",
]
train_labels = [
    ["B-location", "O", "O", "O", "O", "B-location", "O", "B-location"],
    ["O", "O", "O", "B-person", "I-person", "I-person", "O", "O", "B-date"],
]

# The test set of your dataset
test_texts = [
    "John Smith is the best football player in New York",
    "Nowadays , Apple is the richest company .",
    "EU companies rejects German call to boycott British lamb .",
]

test_labels = [
    [
        "B-person",
        "I-person",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "B-location",
        "I-location",
    ],
    ["O", "O", "B-organization", "O", "O", "O", "O", "O"],
    [
        "B-organization",
        "I-organization",
        "O",
        "B-misc",
        "O",
        "O",
        "O",
        "B-misc",
        "O",
        "O",
    ],
]


def from_iob_to_tags(text, labels):
    """
    Create in-text annotated examples using the IOB labeling.
    """

    def remove_iob_prefix(label):
        return label[2:]

    tokens = []
    words = text.split()
    for i in range(len(words)):
        word = words[i]
        label = labels[i]
        if label.startswith("B-"):
            tokens += [f"<{remove_iob_prefix(label)}>", word]
            if i + 1 >= len(words) or labels[i + 1] == "O":
                tokens.append(f"</{remove_iob_prefix(label)}>")
        elif label.startswith("I-"):
            tokens.append(word)
            if i + 1 >= len(words) or labels[i + 1] == "O":
                tokens.append(f"</{remove_iob_prefix(label)}>")
        else:
            tokens.append(word)

    return f"<response> {' '.join(tokens)} </response>"


print(from_iob_to_tags(train_texts[0], train_labels[0]))
# > <response> <location> Moncada </location> is a city near <location> Valencia </location> in <location> Spain </location> </response>

# Let's build the annotated examples
train_annotated_texts = [
    from_iob_to_tags(text, labels_)
    for text, labels_ in zip(train_texts, train_labels)
]
test_annotated_texts = [
    from_iob_to_tags(text, labels_)
    for text, labels_ in zip(test_texts, test_labels)
]

# Let's build the dataset
test_dataset = Dataset.from_dict(
    {"text": test_texts, "annotated_text": test_annotated_texts}
)
few_shot_dataset = Dataset.from_dict(
    {"text": train_texts, "annotated_text": train_annotated_texts}
)
dataset = DatasetDict({"few_shot": few_shot_dataset, "test": test_dataset})

# And save it in a HuggingFace repo to be easily used later
dataset.push_to_hub("jogonba2/ner-lm-eval-harness")
```

We have our dataset for Named Entity Recognition already prepared to be used through lm-evaluation-harness 🥳

## Creating tasks for IOB datasets

The configuration follows the standard format of any other task within lm-evaluation-harness. However, for sequence labeling tasks, it is crucial to include the following fields:

1. `fewshot_split`: This refers to the dataset split prepared in the previous section, which contains a few annotated examples.

2. `description`: This provides the instruction for the task. One essential requirement is to specify the *'tags you can use'*, ensuring that all relevant tags from the dataset are included. Additionally, it is highly recommended to:

    - Clearly define the task, e.g., "Perform named entity recognition by enclosing entity chunks within appropriate tags."

    - Specify delimiters for the final response, e.g., "Wrap your final response within the <response> ... </response> tags." This helps in accurately extracting the final answer and signaling the LLM when to terminate the generation.

3. `metric_list`: Here, the seqeval evaluation suite is used, and the `is_iob` argument must be set to true if the task involves IOB labeling.

For new tasks, it is advisable to reuse the following configuration while adapting the description accordingly for a more precise task definition:

```yaml
task: test_ner
tag:
  - ner
dataset_path: jogonba2/ner-lm-eval-harness
output_type: generate_until
test_split: test
fewshot_split: few_shot
description: 'You must write the provided text by wrapping chunks with entity tags. The entity tags you can use are: <person>, <location>, <organization>, and <misc>. Do not forget to open and close the tag for each entity chunk. Wrap your final response between tags <response> ... </response>.\n\n# Task\n'
doc_to_text: 'Text: {{text}}\nResponse:'
fewshot_delimiter: "\n"
target_delimiter: " "
doc_to_target: annotated_text
num_fewshot: 2
generation_kwargs:
  max_gen_toks: 512
  until:
  - "</response>"
metric_list:
  - metric: seqeval
    higher_is_better: true
    is_iob: true
metadata:
  version: 1.0
```

# Handling word-level tagging

## Build a dataset for tagging

Here, each word is marked with tags. For instance, `<DET> the </DET> <NOUN> dog </NOUN> <VERB> is </VERB> <ADJ> red </ADJ>`.
As with IOB scheme, these in-text annotated examples are then our labels.

> [!IMPORTANT]  
> Note that we are using tags to annotate within the text. So, if your text contain this kind of tags, this method will not work well. It is highly recommended to remove these characters from your dataset for tagging tasks.

```python
# A few training examples from your dataset to elicit the format we expect
train_texts = [
    "Moncada is a city near Valencia in Spain",
    "The football player Miguel Tendillo Valencia comes in July",
]
train_labels = [
    ["A", "B", "C", "D", "A", "B", "C", "D"],
    ["A", "A", "B", "A", "B", "C", "D", "E", "F"],
]

# The test set of your dataset
test_texts = [
    "John Smith is the best football player in New York",
    "Nowadays , Apple is the richest company .",
    "EU companies rejects German call to boycott British lamb .",
]

test_labels = [
    ["A", "C", "A", "B", "D", "E", "F", "A", "B", "C"],
    ["A", "A", "B", "C", "D", "E", "F", "F"],
    ["A", "A", "B", "C", "D", "E", "F", "F", "A", "B"],
]


def from_non_iob_to_tags(text, labels):
    """
    Create in-text annotated examples using the labeling
    """
    tokens = []
    words = text.split()
    for word, label in zip(words, labels):
        tokens.append(f"<{label}> {word} </{label}>")
    return f"<response> {' '.join(tokens)} </response>"


print(from_non_iob_to_tags(train_texts[0], train_labels[0]))
# > <response> <A> Moncada </A> <B> is </B> <C> a </C> <D> city </D> <A> near </A> <B> Valencia </B> <C> in </C> <D> Spain </D> </response>

# Let's build the annotated examples
train_annotated_texts = [
    from_non_iob_to_tags(text, labels_)
    for text, labels_ in zip(train_texts, train_labels)
]
test_annotated_texts = [
    from_non_iob_to_tags(text, labels_)
    for text, labels_ in zip(test_texts, test_labels)
]

# Let's build the dataset
test_dataset = Dataset.from_dict(
    {"text": test_texts, "annotated_text": test_annotated_texts}
)
few_shot_dataset = Dataset.from_dict(
    {"text": train_texts, "annotated_text": train_annotated_texts}
)
dataset = DatasetDict({"few_shot": few_shot_dataset, "test": test_dataset})

# And save it in a HuggingFace repo
dataset.push_to_hub("jogonba2/tagging-lm-eval-harness")
```

We have our dataset for word-level tagging already prepared to be used through lm-evaluation-harness 🥳

## Creating a task for tagging datasets

It is defined in the same way than for IOB datasets, with a single difference, here `is_iob` must be set to `false`:

```yaml
task: test_tagging
tag:
  - tagging
dataset_path: jogonba2/tagging-lm-eval-harness
output_type: generate_until
test_split: test
fewshot_split: few_shot
description: 'You must write the provided text by wrapping each word with tags. The tags you can use are: <A>, <B>, <C>, <D>, <E>, <G>, and <F>. Do not forget to open and close the tag for each word. Wrap your final response between tags <response> ... </response>.\n\n# Task\n'
doc_to_text: 'Text: {{text}}\nResponse:'
fewshot_delimiter: "\n"
target_delimiter: " "
doc_to_target: annotated_text
num_fewshot: 2
generation_kwargs:
  max_gen_toks: 512
  until:
  - "</response>"
metric_list:
  - metric: seqeval
    higher_is_better: true
    is_iob: false
metadata:
  version: 1.0
```

# References

- Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, & Guoyin Wang. (2023). [GPT-NER: Named Entity Recognition via Large Language Models.](https://arxiv.org/abs/2304.10428)

- Naguib, M., Tannier, X., & Nevéol, A. (2024). [Few-shot clinical entity recognition in English, French and Spanish: masked language models outperform generative model prompting](https://aclanthology.org/2024.findings-emnlp.400.pdf). In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 6829–6852). Association for Computational Linguistics.

- Yan Hu, Qingyu Chen, Jingcheng Du, Xueqing Peng, Vipina Kuttichi Keloth, Xu Zuo, Yujia Zhou, Zehan Li, Xiaoqian Jiang, Zhiyong Lu, Kirk Roberts, & Hua Xu. (2024). [Improving Large Language Models for Clinical Named Entity Recognition via Prompt Engineering.](https://arxiv.org/pdf/2303.16416)

- Mingchen Li, & Rui Zhang. (2024). [How far is Language Model from 100% Few-shot Named Entity Recognition in Medical Domain.](https://arxiv.org/pdf/2307.00186)

- Yan, F., Yu, P., & Chen, X. (2024). [LTNER: Large Language Model Tagging for Named Entity Recognition with Contextualized Entity Marking](https://arxiv.org/pdf/2404.05624). In Pattern Recognition: 27th International Conference, ICPR 2024, Kolkata, India, December 1–5, 2024, Proceedings, Part XIX (pp. 399–411). Springer-Verlag.

- Laskar, M., Bari, M., Rahman, M., Bhuiyan, M., Joty, S., & Huang, J. (2023). [A Systematic Study and Comprehensive Evaluation of ChatGPT on Benchmark Datasets](https://aclanthology.org/2023.findings-acl.29.pdf). In Findings of the Association for Computational Linguistics: ACL 2023 (pp. 431–469). Association for Computational Linguistics.

- Machado, M., & Ruiz, E. (2024). [Evaluating large language models for the tasks of PoS tagging within the Universal Dependency framework](https://aclanthology.org/2024.propor-1.46/). In Proceedings of the 16th International Conference on Computational Processing of Portuguese - Vol. 1 (pp. 454–460). Association for Computational Lingustics.

- Stussi, E., & Ströbel, P. (2024). [Part-of-Speech Tagging of 16th-Century Latin with GPT. In Proceedings of the 8th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage](https://aclanthology.org/2024.latechclfl-1.18.pdf), Social Sciences, Humanities and Literature (LaTeCH-CLfL 2024) (pp. 196–206). Association for Computational Linguistics.
