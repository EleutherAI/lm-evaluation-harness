import os
import re
import collections


class ExitCodeError(Exception):
    pass


def sh(x):
    if os.system(x):
        raise ExitCodeError()


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = args_string.split(",")
    args_dict = {}
    for arg in arg_list:
        k, v = arg.split("=")
        args_dict[k] = v
    return args_dict

def join_iters(iters):
    for iter in iters:
        yield from iter


def chunks(iter, n):
    arr = []
    for x in iter:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []
    
    if arr: yield arr

def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)
    
    return list(res.values())

def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace("\" ", "\"")
    string = string.replace(" \"", "\"")
    string = re.sub(r" (['.,])", r"\1", string)
    return string


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [
            ([y[0] for y in x], x[0][1]) for x in arr
        ]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr
        
    
    def get_reordered(self):
        return [x[1] for x in self.arr]
    
    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds: 
                res[ind] = v
                cov[ind] = True
        
        assert all(cov)
        
        return res