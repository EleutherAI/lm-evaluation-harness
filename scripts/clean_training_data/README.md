janitor.py contains a script to remove benchmark data contamination from training data sets.
It uses the approach described in the [GPT-3 paper](https://arxiv.org/abs/2005.14165).

## Algorithm
1) Collects all contamination text files that are to be removed from training data
2) Filters training data by finding `N`gram matches between the training data
   and any contamination
   1) `N`grams ignore case and punctuation and are split on whitespace.
   2) Matching `N`gram substrings are removed, as is a `window_to_remove` character window around
    the match, splitting the training data into chunks
   3) Any chunks less than `minimum_slice_length` are removed
   4) Training data sets split into more than `too_dirty_cutoff` are considered
    completely contaminated and removed

OpenAI used:
```
ngram_n = 13
window_to_remove = 200
minimum_slice_length = 200
too_dirty_cutoff = 10
```

## Compiling

Janitor can be used as a pure python program, but it is much faster if the ngram
code is run in C++. To compile the C++ code, run

```
pip install pybind11
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) janitor_util.cpp -o janitor_util$(python3-config --extension-suffix)
```

MacOS users: If your compiler isn't linked to Python, you may need to add to the above `-undefined dynamic_lookup`. \
Linux users: If your compiler isn't linked to Python, you may need to follow these steps:
1. Rename the compiled code file to `janitor_util.so`.
2. Before running `import Janitor` in your code, add `sys.path.append("your/relative/path/to/janitor_util.so")` so that Python knows the location of `janitor_util.so`.
