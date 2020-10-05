# NLP generally do not require separately downloading data

#coqa
mkdir -p data/coqa
wget http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json -O data/coqa/coqa-train-v1.0.json
wget http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json -O data/coqa/coqa-dev-v1.0.json

#drop
mkdir -p data/drop
wget https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip -O data/drop.zip
unzip data/drop.zip -d data/drop
rm data/drop.zip
mv data/drop/drop_dataset/* data/drop
rm -rf data/drop/drop_dataset
