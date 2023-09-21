class ContextSampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        labeled_examples = (
            self.fewshot_delimiter.join(
                [
                    # TODO: is separating doc_to_text and doc_to_target by one space always desired?
                    (
                        self.doc_to_text(doc)
                        if (
                            self.config.doc_to_choice is None
                            or type(self.doc_to_text(doc)) is str
                        )
                        else self.doc_to_choice(doc)[self.doc_to_text(doc)]
                    )
                    + self.target_delimiter
                    + (
                        str(self.doc_to_target(doc)[0])
                        if type(self.doc_to_target(doc)) is list
                        else self.doc_to_target(doc)
                        if (
                            self.config.doc_to_choice is None
                            or type(self.doc_to_target(doc)) is str
                        )
                        else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
                    )
                    for doc in selected_docs
                ]
            )
            + self.fewshot_delimiter
        )

        return labeled_examples

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)


class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(
            self.docs
        ), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )


# TODO: how should we do design here? might be better to have a single sampler and pass more kwargs at init.
# Depends what's easier for new user to add own functionality on top of

# types of sampler:
# - class-balanced, randomly shuffled
# - class-balanced, one particular set of fewshot examples for all evaled instances
# - hand-specify number of fewshot examples per class?
# - random, varies per example (check that this is curr. default in old repo)
# - random, unified per example
# - enforce a specific fixed fewshot string! (or should we not use this, in favor of including it in prompt template directly)


# - user-specified doc indices to restrict fewshot doc options to
# - user specifies split to use for drawing fewshot instances (TODO: manually prevent this from being same split you eval!)
# - user specifies a prepended "description"/string to add in front of the (prompted) input

# - user specifies a location to draw fewshot samples from? DO THIS IN TASK CLASS
