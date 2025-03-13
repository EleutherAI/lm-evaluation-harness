import re

# xx = [('Naija', 'PROPN'), ('don', 'AUX'), ('carry', 'VERB'), ('knock', 'VERB'), ('give', 'VERB'), ('FG', 'NOUN'), (',', 'PUNCT'), ('and', 'CCONJ'), ('oda', 'PRON'), ('politics', 'NOUN'), ('concerning', 'ADJ'), ('di', 'DET'), ('kilin', 'NOUN')]
# tuple_match = re.findall(r"\(\s*'[^']*',\s*'([^']*)'\s*\)", xx)
# if tuple_match:
#     print(f"tuple_match: {tuple_match}")

# from itertools import chain
# from sklearn.metrics import accuracy_score
# unzipped_list = [(['CCONJ', 'PRON', 'VERB', 'SCONJ', 'PRON', 'VERB', 'PRON', 'ADP', 'VERB', 'ADP', 'ADP', 'DET', 'NOUN', 'PUNCT'], ['ADJ', 'AUX', 'VERB', 'NOUN', 'VERB', 'PROPN', 'PUNCT', 'CCONJ', 'DET', 'NOUN', 'VERB', 'DET', 'NOUN'], ['ADJ', 'NOUN', 'ADV', 'AUX', 'ADP', 'ADP', 'ADJ', 'NOUN', 'PART', 'AUX', 'VERB', 'NOUN', 'ADP', 'NOUN', 'VERB', 'NOUN', 'NOUN', 'CCONJ', 'NOUN', 'NOUN', 'PRON', 'ADP', 'DET', 'NOUN', 'PUNCT']), ([['ADP', 'PRON', 'VERB', 'VERB', 'PRON', 'VERB', 'NOUN', 'ADP', 'VERB', 'ADP', 'ADP', 'PRON', 'NOUN']], [['NOUN', 'VERB', 'VERB', 'VERB', 'VERB', 'PROPN', 'CONJ', 'NOUN', 'NOUN', 'ADJ', 'PROPN', 'NOUN']], [['NOUN', 'PROPN', 'ADV', 'VERB', 'ADP', 'ADJ', 'NOUN', 'PROPN', 'VERB', 'VERB', 'NOUN', 'ADP', 'NOUN', 'VERB', 'ADJ', 'NOUN', 'CONJ', 'ADJ', 'NOUN', 'PROPN', 'NOUN', 'ADJ', 'NOUN']])]
#
# golds, preds = unzipped_list[0], unzipped_list[1]
#
# # Flatten preds' inner lists
# flattened_preds = [list(chain.from_iterable(p)) for p in preds]
# print(f"Golds: {golds} \nPreds: {flattened_preds}")
#
# # Calculate the accuracy for each gold-pred pair
# accuracy_scores = []
# for gold, pred in zip(golds, flattened_preds):
#     # print(f"Golds: {gold} \nPreds: {pred}")
#     # Ensure both lists are of the same length, otherwise truncate to match
#     min_length = min(len(gold), len(pred))
#     gold = gold[:min_length]
#     pred = pred[:min_length]
#
#     # Calculate accuracy for the current pair and add to the list
#     accuracy = accuracy_score(gold, pred)
#     accuracy_scores.append(accuracy)
#
# print(accuracy_scores)
# print(f"Gold: {golds} \nPreds: {preds}")

# accscore = accuracy_score(golds, preds)
# print(accscore)


from datasets import load_dataset
data = load_dataset('facebook/belebele', 'arz_Arab', split='dev')
print(data)
