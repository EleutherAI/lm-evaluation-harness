import collections


Request = collections.namedtuple('Request', ('type', 'args', 'kwargs'))

class RequestFactory:
    def __getattr__(self, attr):
        def fn(*args, **kwargs):
            return Request(attr, args, kwargs)
        return fn


req = RequestFactory()

def MeanAgg(arr):
    return sum(arr) / len(arr)

def MedianAgg(arr):
    return arr[len(arr) // 2]

class ExampleTask(HFTask):
    DATASET_PATH = "example"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        # example

    def validation_docs(self):
        # example

    def test_docs(self):
        # example

    def fewshot_description(self):
        # example

    def doc_to_text(self, doc, include_target=True):
        # example

    def construct_requests(self, doc):
        thing1 = req.logprobs(doc['a'], foo='bar')
        thing2 = req.greedy(doc['b'])
        thing3 = req.somenewrequesttype(doc['c'], flag=True)

        return [thing1, thing2, thing3]

    def process_results(self, doc, results):
        res1, res2, res3 = results

        target = doc['target']

        logprob, _ = res1
        if res2 == target: acc = 1
        else: acc = 0

        weirdmetric = some_weird_thing(res3)

        return {
            'accuracy': (acc, MeanAgg),
            'xentropy': (logprob, MeanAgg),
            'weirdmetric': (weirdmetric, MedianAgg)
        }