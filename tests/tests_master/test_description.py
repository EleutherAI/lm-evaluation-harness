import random
import lm_eval.tasks
import lm_eval.models


def test_description():
    seed = 42
    num_examples = 1
    task_names = ["arc_challenge", "arc_easy"]
    description_dict = {
        "arc_challenge": "Label for the relevant action:\nSentences describing context, with an incomplete sentence trailing answer that plausibly completes the situation.",
        "lambada": "Winograd schema sentence including a either a ___ blank with a missing word, making the pronoun ambiguous, or the same with the word filled in.",
    }

    task_dict = lm_eval.tasks.get_task_dict(task_names)
    for task_name, task in task_dict.items():

        # patch description field in task (# TODO: make this much more cleaned up)
        task._config.description = description_dict[task_name]

        rnd = random.Random()
        rnd.seed(seed)

        if task.has_training_docs():
            docs = task.training_docs()
        elif set == "val" and task.has_validation_docs():
            docs = task.validation_docs()
        elif set == "test" and task.has_test_docs():
            docs = task.test_docs()

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )

        for _, doc in (
            zip(range(num_examples), docs) if num_examples > 0 else enumerate(docs)
        ):
            ctx = task.fewshot_context(
                doc=doc,
                num_fewshot=1,
            )
            assert description in ctx
