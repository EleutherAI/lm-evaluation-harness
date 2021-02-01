import os
from functools import reduce
import operator
from tqdm import tqdm
import json


class ExitCodeError(Exception):
    pass


def sh(x):
    if os.system(x):
        raise ExitCodeError()

def ls(x):
    return [x + '/' + fn for fn in os.listdir(x)]

def lsr(x):
    if os.path.isdir(x):
        return reduce(operator.add, map(lsr, ls(x)), [])
    else:
        return [x]

def fwrite(fname, content):
    with open(fname, 'w') as fh:
        fh.write(content)

def fread(fname):
    with open(fname) as fh:
        return fh.read()

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

class one:
    def __rrshift__(self, other):
        try:
            if isinstance(other, list): 
                assert len(other) == 1
                return other[0]
            return next(other)
        except:
            return None

class join:
    def __init__(self, sep):
        self.sep = sep

    def __rrshift__(self, other):
        if other is None:
            return
        try:
            return self.sep.join(other)
        except:
            return None


Y = object()

def id(x):
    return x

class Reflective:
    def __getattribute__(self, f):
        def _fn(*args, **kwargs):
            return lambda x: x.__getattribute__(f)(*args, **kwargs)
        return _fn
    
    def __getitem__(self, a):
        return lambda x: x[a]
    
    def __mul__(self, other):
        if other == Y:
            def _f(x, y=None):
                if y == None:
                    x, y = x
                
                return x * y
            return  _f
        
        return lambda x: x * other
    
    def __rmul__(self, other):
        if other == Y:
            def _f(x, y=None):
                if y == None:
                    x, y = x
                
                return y * x
            return  _f
        
        return lambda x: other * x
    
    def __add__(self, other):
        if other == Y:
            def _f(x, y=None):
                if y == None:
                    x, y = x
                
                return x + y
            return  _f
        
        return lambda x: x + other
    
    def __radd__(self, other):
        if other == Y:
            def _f(x, y=None):
                if y == None:
                    x, y = x
                
                return y + x
            return  _f
        
        return lambda x: other + x

# (b -> a -> b) -> b -> [a] -> b
def foldl(f, init, arr):
    curr = init
    for elem in arr:
        curr = f(curr, elem)
    return curr

# (a -> b -> b) -> b -> [a] -> b
def foldr(f, init, arr):
    curr = init
    for elem in arr[::-1]:
        curr = f(elem, curr)
    return curr


def comp(*fs):
    if len(fs) == 1:
        return fs[0]
    
    def _f(x):
        for f in fs[::-1]:
            x = f(x)
    
        return x
    return _f


X = Reflective()
