export PYTHONPATH=$PWD
python3 scripts/clean_training_data/generate_13_grams.py \
        -dir /fsx/polyglot/massivetext_large_data/ \
        -sdir /fsx/lime12/ngram_train2/ -n 13 -buckets 500
