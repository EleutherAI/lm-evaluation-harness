import metrics
import sys

def read_output(filename, metric_name=None):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # read only lines that start with '['
    lines = [line for line in lines if line.startswith('[')]
    # split by '] ['
    lines = [line.split('] [') for line in lines]
    # restore '[' and ']'
    lines = [[line[0] + ']', '[' + line[1]] for line in lines]
    # load the lists
    lines = [[eval(line[0]), eval(line[1])] for line in lines]
    # get the metric function from metrics
    if metric_name is None:
        metric_name = 'rounded_acc'
    metric = getattr(metrics, metric_name)
    results = [metric(line[0], line[1]) for line in lines]
    return sum(results) / len(results)


# usage: python read_output.py <filename> [<metric_name>]
print(read_output(*sys.argv[1:]))