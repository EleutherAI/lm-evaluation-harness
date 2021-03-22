from lm_eval import tasks
from pytablewriter import MarkdownTableWriter

writer = MarkdownTableWriter()
writer.headers = ["Task Name", "Train", "Val", "Test", "Metrics"]

values = []

def chk(tf):
    if tf:
        return '✓'
    else:
        return ' '

for tname, Task in tasks.TASK_REGISTRY.items():
    task = Task()

    values.append([tname,chk(task.has_training_docs()),chk(task.has_validation_docs()),chk(task.has_test_docs()),', '.join(task.aggregation().keys())])

writer.value_matrix = values

print(writer.dumps())