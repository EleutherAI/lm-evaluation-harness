# Permutation Composition Benchmark

### Paper

Title: `The Illusion of State in State-Space Models`

Paper Link: https://arxiv.org/abs/2404.08819

Abstract: State-space models (SSMs) have emerged as a potential alternative architecture for building large language models (LLMs) compared to the previously ubiquitous transformer architecture. One theoretical weakness of transformers is that they cannot express certain kinds of sequential computation and state tracking (Merrill & Sabharwal, 2023), which SSMs are explicitly designed to address via their close architectural similarity to recurrent neural networks (RNNs). But do SSMs truly have an advantage (over transformers) in expressive power for state tracking? Surprisingly, the answer is no. Our analysis reveals that the expressive power of SSMs is limited very similarly to transformers: SSMs cannot express computation outside the complexity class TC0. In particular, this means they cannot solve simple state-tracking problems like permutation composition. It follows that SSMs are provably unable to accurately track chess moves with certain notation, evaluate code, or track entities in a long narrative. To supplement our formal analysis, we report experiments showing that Mamba-style SSMs indeed struggle with state tracking. Thus, despite its recurrent formulation, the "state" in an SSM is an illusion: SSMs have similar expressiveness limitations to non-recurrent models like transformers, which may fundamentally limit their ability to solve real-world state-tracking problems.

Dataset: https://huggingface.co/datasets/BeeGass/Group-Theory-Collection

Dataset Generator: https://github.com/BeeGass/Group-Dataset-Generator

### Citation

```bibtex
@inproceedings{merrill2024illusion,
  title={The Illusion of State in State-Space Models},
  author={Merrill, William and Petty, Jackson and Sabharwal, Ashish},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year={2024},
  eprint={2404.08819},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Why this benchmark, and what it measures

This section is written for readers with no background in group theory. It
explains what "state tracking" means, why permutation composition is a clean
way to measure it, and what the TC0/NC1 split in this benchmark is for.

### State tracking, in plain language

State tracking is the ability to carry a single evolving answer through a long
sequence of updates, where every update depends on the result of the previous
one.

A concrete example. Three cups sit on a table in positions 1, 2, 3, and a ball
is under the cup in position 1. Someone performs a long series of swaps:

```text
swap positions 1 and 2
swap positions 2 and 3
swap positions 1 and 3
... (hundreds more swaps)
where is the ball?
```

There is no shortcut. You cannot answer by counting how many swaps mention
position 1, or by looking at the last few swaps, or by any fixed-size summary
of the sequence that ignores order. You have to maintain the current
arrangement and apply each swap to it in turn. That is state tracking: the
work is inherently sequential, and the amount of sequential work grows with the
length of the input.

The same shape shows up whenever the answer is "the current value after all the
updates":

- a variable that is reassigned as it is passed through a chain of functions;
- the position of the pieces on a chess board after a series of moves;
- which claims are in force at a given point in a proof, after assumptions have
  been introduced and discharged;
- who is holding which object at the end of a long narrative.

### Why permutation composition

A permutation is just a rearrangement, exactly like one of the cup swaps above.
Composing two permutations means "do the first rearrangement, then the second",
and the result is again a rearrangement. A permutation group is a fixed,
finite set of rearrangements that is closed under composition.

That gives an unusually clean probe:

- **The task is trivial to state.** Each item is a list of permutation IDs from
  one group, and the answer is the ID of their composed product.
- **The answer is exactly checkable.** The composition is computed from the
  group's multiplication table, so there is a single correct label, no grader
  and no partial credit.
- **Difficulty is one number.** Sequence length is the only knob. This
  benchmark sweeps 5 to 500 in steps of 5 and reports accuracy at each length,
  so the result is a curve, not a single score. Where the curve falls off is
  the interesting quantity.
- **The prior over answers is flat.** Composing uniformly random group elements
  gives a uniformly random product, so a model that ignores the input cannot do
  better than 1/|G|, and there is no surface heuristic to latch onto.
- **The formal difficulty is known.** This is the part the split below is
  about.

The composition here follows the standard mathematical convention and is read
right to left: `p_n ∘ ... ∘ p_2 ∘ p_1` means apply `p_1` first.

### TC0 versus NC1, and why the split matters

TC0 and NC1 are complexity classes for circuits, and they are the right
vocabulary here because a fixed transformer or SSM running over a sequence
behaves like a circuit whose *depth does not grow with the input length*.

- **TC0** is what constant-depth circuits can compute: a fixed number of layers
  of unbounded fan-in majority/threshold gates. Counting, summing, comparing
  and arithmetic all live here. Crucially, so do transformers and SSMs: both
  are provably contained in (uniform) TC0.
- **NC1** allows depth that grows logarithmically with the input length. It
  strictly contains everything in TC0 that we know how to do, and TC0 is
  contained in NC1. Whether that containment is strict is a long-standing open
  problem, but TC0 ≠ NC1 is the near-universal expectation.

Group theory maps directly onto that boundary, and this is the reason to use
groups rather than some ad-hoc puzzle:

- If the group is **solvable** (all the cyclic, dihedral, quaternion,
  elementary abelian, Klein, Frobenius groups here, plus `S3`, `S4`, `A3`,
  `A4`), composing a sequence of its elements is computable in constant depth.
  For a cyclic group it is literally modular addition. These tasks sit inside
  TC0, so a constant-depth model is not ruled out from solving them.
- If the group is **non-solvable**, the picture changes completely.
  Barrington's theorem says the word problem for any finite non-solvable group
  is complete for NC1. `A5`, the alternating group on five elements and the
  smallest non-solvable group, is the canonical case. Composition in `A5`
  therefore lies in TC0 only if TC0 = NC1. Under the standard assumption that
  it does not, no fixed-depth transformer or SSM can solve `A5` composition for
  arbitrary sequence lengths, no matter how wide it is or how much data it
  sees.

That is what makes the split a *separation* rather than a difficulty ranking.
The TC0 groups and the NC1 groups are the same task, the same prompt format,
the same answer space and the same length sweep. The only thing that changes is
whether the underlying algebra is solvable. A model that degrades on both is
simply bad at long sequences. A model that holds up on the TC0 groups and falls
off on the NC1 groups is showing the boundary the theory predicts.

Naming caveat: the task names are labels for which side of that boundary the
group sits on. "TC0 group" here means "composition in this group is in TC0";
it is not a claim that any particular model reaches TC0's limits.

### Why this matters for real models

The practical version of the question is: can a model follow a variable through
a series of transformations?

That is the same capability behind tracking how a value changes as it is passed
through a chain of functions, working out where the pieces are on a chess board
after a series of turns, and following how an argument is set up, supported and
discharged across a proof. The paper above makes the same point in the other
direction: the TC0 bound is what implies that these architectures cannot
reliably track chess moves in a move-only notation, evaluate code, or track
entities across a long narrative.

For attention-based models there are three practical axes that buy state-
tracking ability, and they scale differently against sequence length:

- **depth** - more layers means more sequential composition steps per forward
  pass;
- **width** - more capacity per layer, which does not by itself add sequential
  depth; and
- **chain of thought** - emitting intermediate results and reading them back,
  which converts serial depth into generated tokens and is the one axis that
  escapes the fixed-depth bound.

The length sweep in this benchmark is what makes those axes visible: the point
at which accuracy collapses moves differently depending on which axis you
scale.

A useful lecture-length introduction to this material is Will Merrill's talk on
the paper: https://youtu.be/4-VXe1yPDjk

### Quick start

```bash
# smallest useful signal: one solvable group and one non-solvable group,
# 200 items each, on a small local model
lm_eval --model hf \
  --model_args pretrained=EleutherAI/pythia-160m \
  --tasks s4_composition,a5_composition \
  --batch_size 8 \
  --limit 200
```

Compare the reported per-length metrics (`"5"`, `"10"`, ..., `"500"`) between
the two tasks: `s4` is solvable, `a5` is not.

The full benchmark is large. Each group's `test` split is roughly 20-70MB and
the 94 groups together are about 3GB, so start with a couple of tasks before
running `permutation_groups`.

### Overview

The Permutation Composition Benchmark evaluates a model's ability to perform permutation composition, a key measure of its state-tracking and symbolic reasoning capabilities. The benchmark consists of 94 permutation groups divided into two computational complexity classes based on their algebraic structure:

- **TC⁰** (72 groups): Solvable groups that can be recognized by threshold circuits of constant depth
- **NC¹** (22 groups): Non-solvable groups requiring logarithmic depth circuits

Each task evaluates models on composing sequences of permutations of varying lengths (5 to 500 elements in increments of 5), providing 100 different performance metrics per group. The composition follows the standard mathematical convention (right-to-left): p_n ∘ ... ∘ p_2 ∘ p_1.

### Groups and Tasks

#### Groups

* `permutation_groups`: All 94 permutation groups (includes both TC⁰ and NC¹)
* `tc0_groups`: All 72 TC⁰ (solvable) groups
* `nc1_groups`: All 22 NC¹ (non-solvable) groups

#### Individual Tasks

Each group has a corresponding task named `{group_name}_composition`. Tasks are categorized as follows:

**TC⁰ Groups (Solvable) - 72 groups:**
- **Symmetric**: `s3_composition`, `s4_composition`
- **Alternating**: `a3_composition`, `a4_composition`
- **Cyclic**: `c2_composition` through `c30_composition` (29 groups)
- **Dihedral**: `d3_composition` through `d20_composition` (18 groups)
- **Quaternion**: `q8_composition`, `q16_composition`, `q32_composition`
- **Frobenius**: `f20_composition`, `f21_composition`
- **Klein four-group**: `v4_composition`
- **Elementary abelian**:
  - Z₂ᵏ: `z2_1_composition` through `z2_5_composition`
  - Z₃ᵏ: `z3_1_composition` through `z3_4_composition`
  - Z₅ᵏ: `z5_1_composition` through `z5_4_composition`
- **Projective Special Linear**: `psl2_2_composition`, `psl2_3_composition`

**NC¹ Groups (Non-Solvable) - 22 groups:**
- **Symmetric**: `s5_composition` through `s9_composition`
- **Alternating**: `a5_composition` through `a9_composition`
- **Projective Special Linear**:
  - PSL(2,q): `psl2_4_composition`, `psl2_5_composition`, `psl2_7_composition`, `psl2_8_composition`, `psl2_9_composition`, `psl2_11_composition`
  - PSL(3,q): `psl3_2_composition`, `psl3_3_composition`, `psl3_4_composition`, `psl3_5_composition`
- **Mathieu**: `m11_composition`, `m12_composition`

### Usage

```bash
# Evaluate on all 94 groups
lm_eval --model hf --model_args pretrained=model_name --tasks permutation_groups

# Evaluate on TC⁰ groups only (72 groups)
lm_eval --model hf --model_args pretrained=model_name --tasks tc0_groups

# Evaluate on NC¹ groups only (22 groups)
lm_eval --model hf --model_args pretrained=model_name --tasks nc1_groups

# Evaluate on specific individual groups
lm_eval --model hf --model_args pretrained=model_name --tasks s3_composition,a5_composition,d10_composition

# Limit number of examples per task (useful for testing)
lm_eval --model hf --model_args pretrained=model_name --tasks permutation_groups --limit 100
```

### Task Format

Each task uses a loglikelihood evaluation format where the model must predict the correct permutation ID given a sequence of permutations to compose.

**Input Format:**
```
You are given a sequence of permutations from the group {group_name}, identified by their integer IDs. Your task is to compute their composed product.

The composition must be performed sequentially from right to left, following the standard mathematical convention (p_n ∘ ... ∘ p_2 ∘ p_1).

Sequence: 3 1 4 2 5

Question: What is the single integer ID of the final composed permutation? Your response must be only the integer.

Answer:
```

**Expected Output:** A single integer representing the composed permutation ID.

The prompt is scored as a `loglikelihood` continuation, so `doc_to_text` ends
at `Answer:` with no trailing whitespace and `doc_to_target` supplies the
separating space (`" {{target}}"`).

### Metrics

Each task reports 100 metrics corresponding to sequence lengths from 5 to 500 (in increments of 5):
- Metric names: `"5"`, `"10"`, `"15"`, ..., `"500"`
- Values: Accuracy (0.0-1.0) for samples at that sequence length, or -1 if no samples exist for that length
- Aggregation: Arithmetic mean across all samples at each sequence length

The metrics allow for fine-grained analysis of model performance as sequence length increases, revealing the point at which models fail to maintain accurate state tracking.

### What to expect

These are predictions from the theory, not measurements reported here. No
accuracy numbers for any model are claimed by this task directory.

1. **Fixed-depth models (transformers and SSMs alike)** are contained in
   uniform TC0. On the non-solvable (NC1) groups, accuracy is therefore
   expected to fall towards chance as sequence length grows, and to keep
   falling however much the model is widened. Merrill et al. report exactly
   this behaviour for Mamba-style SSMs.

2. **Chain of thought changes the bound, not the architecture.** Emitting and
   re-reading intermediate products turns serial depth into generated tokens,
   so it is the axis most likely to extend the usable sequence length. That is
   a prediction to test with this benchmark, not a result it contains.

3. **The TC0 groups are the control.** Composition in a solvable group is in
   TC0, so a constant-depth model is not ruled out from solving it. A model
   that fails on both halves is failing at long sequences generally; the
   contrast between the two halves at matched sequence length is the
   informative measurement.

4. **Chance level is 1/|G|.** Interpret accuracy relative to group order, not
   against 0, since the groups here range from order 2 (`c2`) to order 95040
   (`m12`).

### Dataset Details

The benchmark uses the `BeeGass/Group-Theory-Collection` dataset on Hugging Face, which provides:
- Pre-computed permutation sequences for all 94 groups
- Correct composition results verified against group multiplication tables
- Balanced sampling across different sequence lengths
- Each group's dataset can be independently verified using standard group theory algorithms

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation?
    * The paper provides the theoretical framework. The `BeeGass/Group-Theory-Collection` dataset is a concrete implementation following the paper's methodology. Correctness can be verified via standard group multiplication tables.

### Implementation Notes

1. **Output Type**: All tasks use `loglikelihood` output type for consistent evaluation
2. **Data Loading**: Each group has its own dataset loading function in `group_composition_utils.py`
3. **Metric Processing**: Custom metric processing handles sequence length bucketing and aggregation
4. **Memory Efficiency**: Tasks load data on-demand, and only the `test` split of the requested group is downloaded
5. **Generated Configs**: the 94 `*_composition.yaml` files and the three group files are generated, not hand-edited. Edit `generate_tasks.py` and re-run it:

   ```bash
   python lm_eval/tasks/permutation_benchmark/generate_tasks.py
   ```

   `tests/test_permutation_benchmark.py` fails if the committed files drift from the generator's output.

### Changelog

- **v1.0** (2024): Initial release with 94 permutation groups and comprehensive test coverage
