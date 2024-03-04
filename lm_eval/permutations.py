import nltk
import random
import spacy
from nltk.corpus import wordnet
nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
tokenizer = nltk.tokenize.word_tokenize

# This file contains functions for different permutations applied in evaluator.py
# The shuffle is different for each task
def unigram_shuffle(sentence, task):
    words = tokenizer(sentence)  # Tokenize the sentence into words
    # Arc_challenge
    if task == "arc_challenge" and words[0] == "Question":
        words = words[2:-2]  #Don't include "Question:" and "Answer:" in shuffle    
    random.shuffle(words)  # Shuffle the order of words
    return ' '.join(words)  # Join the shuffled words back into a sentence

# Function to shuffle bigrams (pairs of consecutive words) in a sentence
def bigram_shuffle(sentence):
    words = tokenizer(sentence)  # Tokenize the sentence into words
    bigrams = list(nltk.bigrams(words[2:-2]))  # Create bigrams
    random.shuffle(bigrams)  # Shuffle the order of bigrams
    shuffled_words = [word for bigram in bigrams for word in bigram]  # Flatten back to words
    return ' '.join(shuffled_words)  # Join the shuffled words back into a sentence

# Function to shuffle trigrams (groups of three consecutive words) in a sentence
def trigram_shuffle(sentence):
    words = tokenizer(sentence)  # Tokenize the sentence into words
    trigrams = list(nltk.trigrams(words[2:-2]))  # Create trigrams
    random.shuffle(trigrams)  # Shuffle the order of trigrams
    shuffled_words = [word for trigram in trigrams for word in trigram]  # Flatten back to words
    return ' '.join(shuffled_words)  # Join the shuffled words back into a sentence

# Hendrycks dataset adds the multiple choices to the question
# Have to shuffle everything before option A.
def hendrycks_unigram_shuffle(sentence):
    # Find the index of "A" in the sentence
    index_of_A = sentence.find("A.")
    words_before_A = sentence[:index_of_A]
    words_after_A = sentence[index_of_A:]
    words_before_A = tokenizer(words_before_A)
    # Shuffle the words before "A"
    print(index_of_A)
    random.shuffle(words_before_A)
    # Combine the shuffled and unshuffled parts of the sentence
    shuffled_sentence = ' '.join(words_before_A) + '\n' + words_after_A
    return shuffled_sentence

# Replace verbs with antonyms, May extend this to accept a POS as an argument
def getSynonym(word):
    synsets = wordnet.synsets(word, pos=wordnet.VERB)
    synonyms = []
    # Check if there are any synsets available
    if synsets:
        # Get the first synset
        key = synsets[0]
        # Get synonyms
        synonyms = [str(lemma.name()) for lemma in key.lemmas()]
        # Get hypernyms and add them to the list
        for hypernym in key.hypernyms():
            synonyms.extend([str(lemma.name()) for lemma in hypernym.lemmas()])
    return synonyms

# Extract verbs from the prompt and replace them
def verbSynonyms(words):
    doc = nlp(words)
    text = []
    for token in doc:
        if token.pos_ == 'VERB':
            # Get the synonyms
            synonyms = getSynonym(token.text)
            if len(synonyms) > 1:
                text.append(synonyms[1])
            else:
                text.append(token.text)
        else:
            text.append(token.text)
    return " ".join(text)