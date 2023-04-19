from dataclasses import dataclass, field

@dataclass
class Instance:
    request_type: str = None # TODO: make this an enum?
    doc: dict = None
    arguments: tuple = None
    id_: int = None
    metadata: tuple = None # TODO: better typehints here
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    task_name: str = None
    doc_id: str = None
    repeats: str = None

    def __post_init__(self):
        self.task_name, self.doc_id, self.repeats = self.metadata
     
    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)

# import abc

# class Instance(abc.ABC):
#     """
#     A class used to bind together all necessary information and metadata for 
#     running forward pass of a model on a specific datapoint. 

#     """

#     # all Instance subclasses have an attribute which is the name of the LM() class function they call to get outputs.
#     request_type = None

#     def __init__(self, doc, arguments=None, id_=None, metadata=("", None, None)):

#         self.doc = doc # store the document which we're using. this is a dict
#         self.arguments = arguments

#         # need: task name, doc idx, num. repeats
#         self.task_name, self.doc_id, self.repeats = metadata
#         # id_ = idx within a doc's requests
#         self.id_ = id_

#         # handle repeats internally. should be able to run K times on exact same input/output pair
#         # self.repeats = repeats
        
#         # list containing the returns from each call of the model on this particular set of arguments.
#         self.resps = []
#         # filtered_resps should end up a dict, with a different key for each set of filters to apply. calculate results against each key in filtered_resps
#         self.filtered_resps = {}

#         #TODO: add more info as needed for detailed logging

#     def __repr__(self):
#         return f"Req_{self.request_type}{self.args}{self.id_}"

@dataclass
class LoglikelihoodInstance(Instance):

    request_type: str = "loglikelihood"

@dataclass
class RollingLoglikelihoodInstance(Instance):

    request_type: str = "loglikelihood_rolling"

@dataclass
class GenerationInstance(Instance):

    request_type: str = "greedy_until"
