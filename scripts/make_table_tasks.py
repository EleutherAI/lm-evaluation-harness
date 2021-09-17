from lm_eval import tasks
from pytablewriter import MarkdownTableWriter

writer = MarkdownTableWriter()
writer.headers = ["Task Name", "Train", "Val", "Test","Val/Test Docs", "Metrics"]

values = []

def chk(tf):
    if tf:
        return '✓'
    else:
        return ' '

for tname, Task in tasks.TASK_REGISTRY.items():
    task = Task()

    v = [tname,chk(task.has_training_docs()),chk(task.has_validation_docs()),chk(task.has_test_docs()), len(list(task.test_docs() if task.has_test_docs() else task.validation_docs())),', '.join(task.aggregation().keys())]
    print(v)
    values.append(v)

writer.value_matrix = values

print(writer.dumps())