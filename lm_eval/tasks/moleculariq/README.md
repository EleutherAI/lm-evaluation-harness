# MolecularIQ Benchmark Task

MolecularIQ is a fully symbolically verifiable benchmark for assessing reasoning over molecular structures.
It covers three task types -- feature counting, index-based attribution, and constrained generation -- across
six categories of symbolically verifiable molecular features (graph topology, chemistry-typed topology,
composition, chemical perception, functional groups, and synthesis/fragmentation) and three orthogonal
complexity axes (SMILES representation format, molecular complexity, and multitask load).

**Paper**: [MolecularIQ: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs](https://arxiv.org/abs/2601.15279)

**Leaderboard**: [https://huggingface.co/spaces/ml-jku/molecularIQ_leaderboard](https://huggingface.co/spaces/ml-jku/molecularIQ_leaderboard)

**Code**: [https://github.com/ml-jku/moleculariq](https://github.com/ml-jku/moleculariq)

## Dataset

- **Dataset**: `ml-jku/moleculariq-v0.0`
- **Split**: test (5,111 questions from 849 unique molecules)

## Task Types

The benchmark includes three reasoning task types:

1. **Feature Counting** (1,800 questions): Establishes baseline comprehension of molecular graphs.
   - `single_count`: Single property counting (e.g., "How many rotatable bonds?")
   - `multi_count`: Multiple property counting in one question

2. **Index-based Attribution** (1,740 questions): Requires models to ground answers in specific atoms/bonds/substructures.
   - `single_index`: Single index identification (e.g., "Which atoms are part of the Murcko scaffold?")
   - `multi_index`: Multiple index identification

3. **Constrained Generation** (1,571 questions): Generating molecules that satisfy given constraints.
   - `single_constraint_generation`: Single constraint
   - `multi_constraint_generation`: Multiple constraints

## Molecular Feature Categories

- **Graph Topology**: Rings, fused rings, bridgehead atoms, ring size extremes, linear termini, branch points
- **Chemistry-typed Topology**: Aromatic vs. aliphatic rings, heterocycles, saturation, sp3 carbon hybridization, longest carbon chains, stereochemical descriptors
- **Composition**: Carbon, hetero, halogen, heavy, hydrogen atom counts, molecular formula
- **Chemical Perception**: Hydrogen bond donors/acceptors, rotatable bonds, oxidation states
- **Functional Groups**: Alcohols, amines, carboxylic acids, and other standard functional groups
- **Synthesis/Fragmentation**: BRICS fragmentation, template-based reaction prediction, Murcko scaffold extraction

## Complexity Axes

- **Molecular Complexity** (Bertz index): Simple (0-250), Medium (250-1000), Complex (1000+)
- **Multitask Load**: 1, 2, 3, or 5 simultaneous sub-tasks per question
- **SMILES Representation**: Canonical, kekulized, randomized, ring-number enumerated

## Metrics

- **pass_at_1**: Accuracy on first attempt
- **pass_at_3**: Any correct answer in first 3 attempts
- **avg_accuracy**: Average accuracy across all attempts

Scoring uses a binary symbolic verifier; per-instance accuracy is the mean over independent rollouts.

## Available Tasks

| Task | Description |
|------|-------------|
| `moleculariq_pass_at_k` | Raw question only - use with `--system_instruction` or chat models |
| `moleculariq_inline` | Question with inline prompt (instructions + answer format) |

## Usage

### Basic Usage (Raw Question)

For chat models or when using `--system_instruction`:

```bash
lm_eval --model vllm \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_pass_at_k \
    --batch_size auto
```

### With Inline Prompt

For models that need explicit instructions in the prompt:

```bash
lm_eval --model vllm \
    --model_args pretrained=your-model-name \
    --tasks moleculariq_inline \
    --batch_size auto
```

### Quick Test (Limited Samples)

```bash
lm_eval --model vllm \
    --model_args pretrained=facebook/opt-125m \
    --tasks moleculariq_pass_at_k \
    --limit 10 \
    --batch_size auto
```

## Directory Structure

```
moleculariq/
├── __init__.py                    # Package init
├── moleculariq_pass_at_k.yaml     # Task with raw question only
├── moleculariq_inline.yaml        # Task with inline prompt
├── task_processor.py              # Processing hooks (uses moleculariq_core)
├── extractors.py                  # Answer extraction functions
└── README.md
```

## Dependencies

- `moleculariq_core`: Core library for molecular reasoning and reward computation
- `rdkit`: Chemistry toolkit (dependency of moleculariq_core)
- `datasets`: For loading the HuggingFace dataset

Install dependencies:
```bash
pip install moleculariq-core rdkit
```

## Answer Format

Models should return answers in JSON format within answer tags:

```
<answer>{"property_name": value}</answer>
```

For count tasks:
```
<answer>{"ring_count": 2}</answer>
```

For index tasks:
```
<answer>{"carbon_indices": [0, 1, 3]}</answer>
```

For constraint generation:
```
<answer>{"smiles": "CCO"}</answer>
```

## Atom Indexing Convention

Atoms are indexed from 0 to N-1, reading the SMILES string left to right, counting only heavy atoms (non-hydrogen). Examples:

- `"CCO"`: C(0), C(1), O(2)
- `"CC(C)O"`: C(0), C(1), C(2), O(3)
- `"CC(=O)N"`: C(0), C(1), O(2), N(3)

## Citation

```bibtex
@inproceedings{bartmann2026moleculariq,
  title={Molecular{IQ}: Characterizing Chemical Reasoning Capabilities Through Symbolic Verification on Molecular Graphs},
  author={Christoph Bartmann and Johannes Schimunek and Mykyta Ielanskyi and Philipp Seidl and G{\"u}nter Klambauer and Sohvi Luukkonen},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=RqwEzZqMFv}
}
```
