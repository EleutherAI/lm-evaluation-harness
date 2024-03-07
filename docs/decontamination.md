# Decontamination

## Usage

The provided directory should contain
the ngram files and info.json produced in "Pile Ngram Generation" further down.

```bash
python -m lm_eval \
    --model gpt2 \
    --device 0 \
    --tasks sciq
```

## Background
Downstream evaluations test model generalization, and are less useful when test set data also exists in the training set, referred to as leakage or contamination.

Filtering your training set against the test set is a good first step, however this isn't always possible, as in the case of a new benchmark or one that wasn't considered prior to model training. When training set filtering isn't possible, it is useful to measure the impact of test set leakage by detecting the contaminated test examples and producing a clean version of the benchmark.

The basis for our decontamination procedure can be found in Appendix C of "Language Models are Few-Shot Learners". OpenAI defined a test document as contaminated if any N-gram overlap existed with any training document. They used a range of N values between 8 and 13 depending on dataset, while we just used 13 for simplicity.

## Implementation
Contamination detection can be found in `lm_eval/decontaminate.py` with supporting code in `lm_eval/decontamination/`.

decontaminate.py does the following:
1. Build dictionaries of all ngrams and their corresponding evaluation/document ids.
2. Scan through sorted files containing training set n-grams.
3. If a match is found, the corresponding evaluation/document combinations are marked as contaminated.

`lm_eval/evaluator.py` can then produce a clean version of the benchmark by excluding the results of contaminated documents. For each metric, a clean version will be shown in the results with a "decontaminate" suffix.

This is disabled by default for new tasks, to support decontamination on a task override the "should_decontaminate" and "doc_to_decontamination_query" methods. For more details see the [task guide](task_guide.md).

## Pile Ngram Generation
The relevant scripts can be found in `scripts/clean_training_data`, which also import from
`lm_eval/decontamination/`

1. git clone https://github.com/EleutherAI/lm-evaluation-harness.git
2. pip install -r requirements.txt
3. Download The Pile from [The Eye](https://the-eye.eu/public/AI/pile/train/)
4. Place pile files in "pile" directory under "lm-evaluation-harness" (or create a symlink)
5. Run generate_13_grams.

```bash
export PYTHONHASHSEED=0
python -m scripts/clean_training_data/generate_13_grams \
       -dir path/to/working/directory \
       -n 13 \
       -buckets 500
```

Took approximately 4 days for us. We had the time to wait, but this could be scaled out by doing partial pile scans on multiple instances of this script and merging the relevant buckets. We fixed PYTHONHASHSEED to ensure reproducibility of bucket hashing in case you need to stop and start.

6. Sort the generated 13-grams.
```bash
python -m scripts/clean_training_data/sort_13_gram_buckets \
       -dir path/to/working/directory/output
```

Took approximately 5 days for us. You could speed this up by spreading the files around to different machines and running the sort script before gathering them together.

7. Compress the sorted 13 grams files and place them together with info.json.

This step only takes a few hours.

```bash
python -m scripts/clean_training_data/compress_and_package \
       -dir path/to/working/directory \
       -output path/to/final/directory \
       -procs 8
```
