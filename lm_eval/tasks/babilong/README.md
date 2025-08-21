# Babilong

### Paper

Title: `Babilong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack`

Abstract: `Babilong is a benchmark for evaluating the ability of language models to reason over extremely long contexts (up to 10 million tokens). It focuses on testing models' ability to perform multi-hop reasoning and fact retrieval across documents of varying lengths. The benchmark includes 20 different reasoning tasks inspired by the bAbI dataset, scaled to test performance across context lengths from 1k to 10M tokens.`

Homepage: `https://github.com/booydar/babilong`

### Citation

```
@article{kuratov2024babilong,
    title={Babilong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack},
    author={Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Burtsev, Mikhail},
    journal={arXiv preprint arXiv:2406.10149},
    year={2024}
}
```

### Groups and Tasks

#### Groups

* `babilong`: All Babilong tasks
* `babilong_qa`: Question answering tasks
* `babilong_list`: List processing tasks  
* `babilong_reasoning`: Complex reasoning tasks

#### Tasks

The benchmark includes 20 reasoning tasks at various context lengths:

**QA Tasks (qa1-qa5):**
* `babilong_qa1_single_fact`: Single supporting fact QA
* `babilong_qa2_two_facts`: Two supporting facts QA
* `babilong_qa3_three_facts`: Three supporting facts QA
* `babilong_qa4_two_arg_relations`: Two argument relations
* `babilong_qa5_three_arg_relations`: Three argument relations

**List Tasks (qa6-qa10):**
* `babilong_qa6_yes_no`: Yes/No questions
* `babilong_qa7_counting`: Counting entities
* `babilong_qa8_lists_sets`: Lists and sets operations
* `babilong_qa9_simple_negation`: Simple negation
* `babilong_qa10_indefinite_knowledge`: Indefinite knowledge

**Reasoning Tasks (qa11-qa15):**
* `babilong_qa11_basic_coreference`: Basic coreference resolution
* `babilong_qa12_conjunction`: Conjunction reasoning
* `babilong_qa13_compound_coreference`: Compound coreference
* `babilong_qa14_time_reasoning`: Time-based reasoning
* `babilong_qa15_basic_deduction`: Basic deduction

**Advanced Tasks (qa16-qa20):**
* `babilong_qa16_basic_induction`: Basic induction
* `babilong_qa17_positional_reasoning`: Positional reasoning
* `babilong_qa18_size_reasoning`: Size-based reasoning
* `babilong_qa19_path_finding`: Path finding in graphs
* `babilong_qa20_agents_motivations`: Understanding agent motivations

Each task is available at multiple context lengths:
- 1k, 4k, 16k, 32k, 64k, 128k, 256k, 512k, 1M, 2M, 4M, 10M tokens

### Features

- **Haystack Design**: Each task embeds the relevant information within irrelevant "distractor" text
- **Controlled Complexity**: Tasks progressively increase in reasoning complexity
- **Length Scaling**: Systematic evaluation across 12 different context lengths
- **Multi-hop Reasoning**: Many tasks require connecting multiple pieces of information
- **Fact Tracking**: Tests ability to track entities and facts across long distances

### Evaluation

Tasks are evaluated using exact match accuracy. For each task:
- Models must extract and reason over relevant facts from the long context
- Answer format is typically short (single word or number)
- Performance is measured across different context lengths to assess scaling

### Notes

- Babilong extends the classic bAbI tasks to extreme context lengths
- Tasks are designed to be solvable with perfect accuracy given sufficient context understanding
- The benchmark tests both retrieval and reasoning capabilities
- Includes both English and multilingual versions