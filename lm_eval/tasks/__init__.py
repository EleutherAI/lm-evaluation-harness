import os

from lm_eval.api.task import register_yaml_task

from .vanilla import *

# we want to register all yaml tasks in our .yaml folder.
yaml_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + "yaml"


for yaml in sorted(os.listdir(yaml_dir)):
    yaml = os.path.join(yaml_dir, yaml)
    register_yaml_task(yaml)
