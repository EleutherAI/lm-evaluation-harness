import re

try:
    from rouge_score import rouge_scorer, scoring
except ImportError:
    rouge_scorer, scoring = None, None


class MecabTokenizer:
    def __init__(self) -> None:
        from fugashi import Tagger

        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text):
        """Lower case text, remove punctuation and extra whitespace, etc."""
        import emoji
        import neologdn

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_emoji(text):
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "]+",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub(r"", text)

        text = remove_emoji(text)
        # see neologdn docs for details, but handles things like full/half width variation
        text = neologdn.normalize(text)
        text = white_space_fix(text)
        return text

    def tokenize(self, text):
        return self.tagger.parse(self.normalize_answer(text)).split()


def rouge2(items):
    return items


def rouge2_agg(items):
    if rouge_scorer is None or scoring is None:
        raise RuntimeError("rouge2 dependency is not available")

    tokenizer = MecabTokenizer()

    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    rouge_type = "rouge2"

    # mecab-based rouge
    scorer = rouge_scorer.RougeScorer(
        rouge_types=[rouge_type],
        tokenizer=tokenizer,
    )

    # Acumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()

    return result[rouge_type].mid.fmeasure
