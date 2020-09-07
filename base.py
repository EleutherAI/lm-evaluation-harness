import abc
import random

class LM(abc.ABC):
    @abc.abstractmethod
    def generate(self, context, until):
        pass

    @abc.abstractmethod
    def loglikelihood(self, context, continuation):
        pass


class Dataset(abc.ABC):
    @abc.abstractmethod
    def training_docs(self):
        pass
    
    @abc.abstractmethod
    def validation_docs(self):
        pass
    
    @abc.abstractmethod
    def test_docs(self):
        pass
    
    def fewshot_examples(self, k):
        traindocs = list(self.training_docs())
        random.seed(123)
        random.shuffle(traindocs)

        return traindocs[:k]
    
    @abc.abstractmethod
    def fewshot_description(self):
        pass

    @abc.abstractmethod
    def doc_to_text(self, doc, include_target=True):
        pass
    
    @abc.abstractmethod
    def evaluate(self, docs, lm, provide_description, num_fewshot):
        pass
    