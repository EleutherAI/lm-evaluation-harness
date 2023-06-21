class Sampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None):

        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.delimiter = self.config.delimiter

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
            self.delimiter.join(
                [
                    # TODO: is separating doc_to_text and doc_to_target by one space always desired?
                    self.task.doc_to_text(doc)
                    + self.config.sample_delimiter
                    + self.task.doc_to_target(doc)
                    for doc in selected_docs
                ]
            )
            + self.delimiter
        )

        # only returns the fewshot context! Does not append the document, do this outside the object
        return labeled_examples

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)


class BalancedSampler(Sampler):
    def sample(self, n):
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(Sampler):
    def sample(self, n):
        """ """
        pass


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
