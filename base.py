import abc
import random


class LM(abc.ABC):
    @abc.abstractmethod
    def generate(self, context, max_gen_length):
        """Conditional text generation with an LM

        :param context: str
            Context string for conditional generation
        :param max_gen_length: int
            Maximum number of tokens to generate
        :return: str

        """
        pass

    @abc.abstractmethod
    def loglikelihood(self, context, continuation):
        """Compute log-prob of a generation a continuation from a context

        Assume that the final text will simple be
            context + continuation

        :param context: str
            Context string for conditional generation
        :param continuation: str
            Maximum number of tokens to generate
        :return: float
        """
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string):
        """Constructor method, in case models need additional arguments
        e.g. OpenAI API engine, paths for loading, other params

        :param arg_string: str
            Left up to individual model class to handle

        """
        return cls()


class Dataset(abc.ABC):
    @abc.abstractmethod
    def has_training_docs(self):
        pass
    
    @abc.abstractmethod
    def has_validation_docs(self):
        pass

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


class Registry:
    def __init__(self, registry_name):
        self.registry_name = registry_name
        self.registry = {}

    def register(self, name):
        def register_cls(new_cls):
            if name in self.registry:
                raise ValueError('Cannot register duplicate ({})'.format(self.registry_name, name))
            self.registry[name] = new_cls
            return new_cls
        return register_cls
