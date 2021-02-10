import os
import re


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

def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace("\" ", "\"")
    string = string.replace(" \"", "\"")
    string = re.sub(r" (['.,])", r"\1", string)
    return string