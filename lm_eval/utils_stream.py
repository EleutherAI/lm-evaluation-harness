import os
from functools import reduce
import operator
from tqdm import tqdm
import json

# TODO: phase out utils_stream

class each:
    def __init__(self, f):
        self.f = f

    def __rrshift__(self, other):
        return list(map(self.f, other))

class filt:
    def __init__(self, f):
        self.f = f

    def __rrshift__(self, other):
        return list(filter(self.f, other))

class apply:
    def __init__(self, f):
        self.f = f

    def __rrshift__(self, other):
        return self.f(other)
