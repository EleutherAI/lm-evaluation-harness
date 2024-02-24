import os
import sys

from lm_eval.__main__ import cli_evaluate

current_script_directory = os.path.dirname(os.path.abspath(__file__))
evaluate_directory = os.path.normpath(os.path.join(current_script_directory, '..',  '..', 'evaluate'))
train_directory = os.path.normpath(os.path.join(current_script_directory, '..',  '..', 'train'))
sys.path.insert(0, evaluate_directory) # Insert at the beginning to ensure evaluate/zgeval has priority
sys.path.append(train_directory)

print(evaluate_directory, train_directory)

import based_hf as based_hf

if __name__ == "__main__":
    cli_evaluate()