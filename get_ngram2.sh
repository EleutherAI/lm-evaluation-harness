export PYTHONPATH=$PWD
python3 scripts/clean_training_data/generate_13_grams.py \
        -dir /fsx/kevinai/data/ko/merged_raw/ \
        -sdir /fsx/lime12/ngram_merged_raw -n 13 -buckets 500