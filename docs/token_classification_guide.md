# Token classification

Token classification datasets can be of many different forms, but here we assume that we have a dataset of texts with labels at word-level, either IOB scheme for tasks like Named Entity Recognition or aspect detection, or non-IOB scheme for tasks like POS Tagging.

If your dataset does not meet this, you will have to preprocess it.

Let's start with IOB tagging for Named Entity Recognition

# Working with IOB scheme

## Build a dataset from IOB scheme
We need to prepare our IOB-labeled dataset in a scheme that (i) LLMs can understand and generate, and (ii) is manageable by LM Eval Harness.

We do so, by creating in-text annotated examples, where each chunk is marked with tags. For instance, \<person\> John Smith \</person\> plays football.

These in-text annotated examples are then our labels, and we can reconstruct later the IOB labeling from the annotated examples.

> [!IMPORTANT]  
> Note that we are using tags to annotate within the text. So, if your text contain this kind of tags, this method will not work well. It is highly recommended to remove these characters from your dataset for tagging tasks.

> [!IMPORTANT]  
> It is very recommended for the texts in your dataset to contain separation symbols separated by a whitespace from content words. For instance, "Apple, the company is one of the richest.", can lead to tagging errors since the LLMs typically consider "," or "." as separate tokens. You can use `nltk.word_tokenize` to go from that to "Apple , the company is one of the richest ."

```python
# A few training examples from your dataset to be used as few-shot examples in LM Eval Harness
# Few-shot examples are mandatory for these tasks since we need to elicit the format we expect.
train_texts = ["Moncada is a city near Valencia in Spain", "The football player Miguel Tendillo Valencia comes in July"]
train_labels = [["B-location", "O", "O", "O", "O", "B-location", "O", "B-location"],
                ["O", "O", "O", "B-person", "I-person", "I-person", "O", "O", "B-date"]]

# The test set of your dataset
test_texts = ["John Smith is the best football player in New York",
         "Nowadays , Apple is the richest company .",
         "EU companies rejects German call to boycott British lamb ."]

test_labels = [["B-person", "I-person", "O", "O", "O", "O", "O", "O", "B-location", "I-location"],
          ["O", "O", "B-organization", "O", "O", "O", "O", "O"],
          ["B-organization", "I-organization", "O", "B-misc", "O", "O", "O", "B-misc", "O", "O"]]

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

# And save it in a HuggingFace repo
dataset.push_to_hub("jogonba2/ner-llm-eval-harness", private=True)
```

Here, we have our dataset already prepared to be used through LM Eval Harness.

## Creating a YAML config for IOB datasets
The config is like any other existing metric in LM Eval Harness, but it is very important for token classification tasks to add:

1. `fewshot_split`: the split we prepared in the previous section for few-shot annotated examples
2. `description`: the instruction for your task. Here, there is one mandatory thing you have to add to the description, that is the 'tags you can use', which must include the existing tags in the dataset. It is highly recommended to (i) describe better your task, e.g., "You have to do named entity recognition by wrapping chunks with entity tags", and (ii) specify delimiters for the final response, e.g., "Wrap your final response between tags \<response\> ... \</response\>". The later allows us to properly extract the final answer and to tell the LLM when to stop the generation.
3. `is_iob`: you have to set this argument to `true` for IOB labels as in this case.

For new tasks, it is recommended to reuse the following configuration, changing the `description` accordingly to better define your task:

```yaml
task: ner_iberbench
tag:
  - ner
dataset_path: jogonba2/ner-llm-eval-harness
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
  - metric: iberbench_seqeval
    higher_is_better: true
    is_iob: true
metadata:
  version: 1.0
```

The evaluation metric `iberbench_seqeval` will be applied, which basically calls `seqeval` and returns the mean of `overall_f1` per sample.

# Working with non-IOB scheme

## Build a dataset from non-IOB scheme

Differently from IOB scheme, here, each word is marked with tags. For instance, \<DET\> the \</DET\> \<NOUN\> dog \</NOUN\> \<VERB\> is \</VERB\> \<ADJ\> red \</ADJ\>.
As with IOB scheme, these in-text annotated examples are then our labels.

```python
# A few training examples from your dataset to be used as few-shot examples in LM Eval Harness
# Few-shot examples are mandatory for these tasks since we need to elicit the format we expect.
train_texts = ["Moncada is a city near Valencia in Spain", "The football player Miguel Tendillo Valencia comes in July"]
train_labels = [["A", "B", "C", "D", "A", "B", "C", "D"],
                ["A", "A", "B", "A", "B", "C", "D", "E", "F"]]

# The test set of your dataset
test_texts = ["John Smith is the best football player in New York",
         "Nowadays , Apple is the richest company .",
         "EU companies rejects German call to boycott British lamb ."]

test_labels = [["A", "C", "A", "B", "D", "E", "F", "A", "B", "C"],
          ["A", "A", "B", "C", "D", "E", "F", "F"],
          ["A", "A", "B", "C", "D", "E", "F", "F", "A", "B"]]

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
dataset.push_to_hub("jogonba2/tagging-llm-eval-harness", private=True)
```

Here, we have our dataset already prepared to be used through LM Eval Harness.

## Creating a YAML config for non-IOB datasets

It is defined in the same way than for IOB datasets, with a single difference, here `is_iob` must be set as `False`. Also, remember to remember the LLM to wrap each word 😛:

```yaml
task: tagging_iberbench
tag:
  - tagging
dataset_path: jogonba2/tagging-llm-eval-harness
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
  - metric: iberbench_seqeval
    higher_is_better: true
    is_iob: false
metadata:
  version: 1.0
```

The evaluation metric `iberbench_seqeval` will be applied, which basically calls `seqeval` and returns the mean of `overall_f1` per sample.