import base
import nlp


def yesno(x):
    if x: return 'yes'
    else: return 'no'


def mean(x):
    return sum(x) / len(x)


class BoolQ(base.Dataset):
    def __init__(self):
        self.dataset = nlp.load_dataset('boolq')

    def training_docs(self):
        yield from self.dataset['train']
    
    def validation_docs(self):
        yield from self.dataset['validation']
    
    def test_docs(self):
        return []
    
    def fewshot_examples(self, k):
        traindocs = list(self.training_docs())
        random.seed(123)
        random.shuffle(traindocs)

        return traindocs[:k]
    
    def fewshot_description(self):
        return "Read the following passage and answer the question with a yes or a no."

    def doc_to_text(self, doc, include_target=True):    
        return f"{doc['passage']}\nquestion: {doc['question']}\nanswer: " + (yesno(doc['answer']) if include_target else "")
    
    def evaluate(self, docs, lm, provide_description, num_fewshot):
        acc = []
        for doc in docs:
            ctx = '\n\n'.join(map(doc_to_text, self.fewshot_examples())) + '\n\n'
            ctx += doc_to_text(doc, include_target=False)
            ctx = ((self.fewshot_description() + "\n") if provide_description else "") + ctx

            ans = lm.loglikelihood(ctx, 'yes') > lm.loglikelihood(ctx, 'no')

            acc.append(int(ans == doc['answer']))
    
        return mean(acc)