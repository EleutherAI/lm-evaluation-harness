from itertools import islice

from lm_eval import tasks


ct = 3

for (
    tname,
    Task,
) in tasks.TASK_REGISTRY.items():  # [('record', tasks.superglue.ReCoRD)]:#
    task = Task()

    print("#", tname)
    docs = islice(
        task.validation_docs() if task.has_validation_docs() else task.test_docs(), ct
    )
    print()
    for i in range(ct):
        print()
        doc = next(docs)
        print("**Context**:", "\n```\n" + task.doc_to_text(doc) + "\n```\n")
        print()
        print("**Target**:", "\n```\n" + task.doc_to_target(doc) + "\n```\n")
        print()
