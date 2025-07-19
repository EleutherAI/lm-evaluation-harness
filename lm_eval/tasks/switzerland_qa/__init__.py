from lm_eval.tasks.yaml_task import register_yaml_tasks

# register every *.yaml in this directory (and sub-dirs) as a task
register_yaml_tasks(__name__, globals())