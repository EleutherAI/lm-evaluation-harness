import re


def _missing_module_message(name):
    return f"`{name}` is required for `japanese_leaderboard`, please install `{name}` via pip install lm_eval[japanese_leaderboard] or pip install -e .[japanese_leaderboard]"


try:
    import emoji
    import neologdn
    from fugashi import Tagger
    from rouge_score import rouge_scorer, scoring
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(_missing_module_message(err.name)) from err


class MecabTokenizer:
    def __init__(self) -> None:
        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text):
        """Lower case text, remove punctuation and extra whitespace, etc."""

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_emoji(text):
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001f600-\U0001f64f"  # emoticons
                "\U0001f300-\U0001f5ff"  # symbols & pictographs
                "\U0001f680-\U0001f6ff"  # transport & map symbols
                "\U0001f1e0-\U0001f1ff"  # flags (iOS)
                "\U00002702-\U000027b0"
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
