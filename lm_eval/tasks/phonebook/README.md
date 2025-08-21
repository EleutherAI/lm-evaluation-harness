# Phonebook Lookup

### Paper

Title: `Lost in the Middle: How Language Models Use Long Contexts`

Abstract: `The Phonebook Lookup task is a synthetic benchmark designed to test language models' ability to retrieve specific information from long contexts. The task involves finding a phone number associated with a specific name within a "phonebook" containing hundreds or thousands of entries. This task was introduced as part of research showing that language models often struggle to use information in the middle of long contexts, exhibiting a U-shaped performance curve where they best utilize information at the beginning and end of the input.`

Homepage: `https://github.com/nelson-liu/lost-in-the-middle`

### Citation

```
@article{liu2023lost,
    title={Lost in the Middle: How Language Models Use Long Contexts},
    author={Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
    journal={arXiv preprint arXiv:2307.03172},
    year={2023}
}
```

### Task Description

The Phonebook Lookup task presents models with a long list of name-phone number pairs and asks them to retrieve the phone number for a specific person. The task tests:

1. **Information Retrieval**: Can the model find specific information in a long context?
2. **Position Sensitivity**: How does the position of the target information affect retrieval accuracy?
3. **Distractor Robustness**: Can the model handle many irrelevant entries?

### Groups and Tasks

#### Groups

* `phonebook`: All phonebook lookup tasks
* `phonebook_short`: Short context variants (1K-10K tokens)
* `phonebook_long`: Long context variants (10K-100K+ tokens)

#### Task Variants

* `phonebook_1k`: ~1,000 token context (~50 entries)
* `phonebook_5k`: ~5,000 token context (~250 entries)
* `phonebook_10k`: ~10,000 token context (~500 entries)
* `phonebook_25k`: ~25,000 token context (~1,250 entries)
* `phonebook_50k`: ~50,000 token context (~2,500 entries)
* `phonebook_100k`: ~100,000 token context (~5,000 entries)
* `phonebook_200k`: ~200,000 token context (~10,000 entries)

### Task Format

Each instance contains:
- A list of phonebook entries in the format: "Name: [Full Name], Phone: [Phone Number]"
- A query asking for a specific person's phone number
- The position of the target entry is varied (beginning, middle, end)

Example:
```
Phonebook:
Name: John Smith, Phone: 555-1234
Name: Jane Doe, Phone: 555-5678
... [many more entries] ...
Name: Bob Johnson, Phone: 555-9012

Question: What is Jane Doe's phone number?
Answer: 555-5678
```

### Evaluation

- **Metric**: Exact match accuracy
- **Position Analysis**: Performance is typically analyzed by position (beginning, middle, end)
- **Length Scaling**: Performance across different context lengths

### Key Findings

Research using this task has shown:
1. **U-shaped Curve**: Models perform best when information is at the beginning or end
2. **Middle Degradation**: Significant performance drop for information in the middle
3. **Length Impact**: Performance generally decreases with longer contexts
4. **Model Variation**: Different models exhibit varying degrees of position bias

### Implementation Notes

- Phone numbers are generated randomly but consistently
- Names are selected from a diverse pool to avoid ambiguity
- Each context length has multiple variants with target at different positions
- The task can be extended to test other retrieval patterns

### Variations

Additional variants can test:
- Multiple queries per context
- Ambiguous names requiring disambiguation
- Formatted vs. unformatted phone numbers
- International phone number formats
- Partial name matching
