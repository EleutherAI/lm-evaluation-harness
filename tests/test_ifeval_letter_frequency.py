import random
import string

import pytest


# The IFEval scorers import langdetect, immutabledict and nltk at module scope.
# Those live in the [ifeval] extra, which the unit test workflow does not install.
ifeval_utils = pytest.importorskip(
    "lm_eval.tasks.ifeval.utils",
    reason="IFEval needs the [ifeval] extra: langdetect, immutabledict, nltk",
)

from lm_eval.tasks.ifeval import instructions_registry


LETTER_FREQUENCY = "keywords:letter_frequency"

# The two IFEval prompts whose letter_frequency target is punctuation.
PUNCTUATION_PROMPTS = [
    pytest.param(
        1122,
        "make a tweet for playboy's twitter account without using capital "
        "letters. Include at least 4 hashtags, starting with '#'",
        {"let_relation": "at least", "letter": "#", "let_frequency": 4},
        "#ocean #sun #sand #summer vibes all day",
        "no tags here, just a plain lowercase tweet",
        id="key-1122-hashtags",
    ),
    pytest.param(
        1129,
        'Write a short startup pitch for a new kind of ice cream called "Sunnis '
        'ice cream". The ice cream should be gentle on the stomach. Contain 6 or '
        'more exclamation marks "!" in your response.',
        {"let_relation": "at least", "letter": "!", "let_frequency": 6},
        "Try it! Love it! Share it! Gentle! Creamy! Sunnis!",
        "Sunnis ice cream is gentle on the stomach.",
        id="key-1129-exclamations",
    ),
]


def _make_input(key, prompt, kwargs):
    return ifeval_utils.InputExample(
        key=key,
        instruction_id_list=[LETTER_FREQUENCY],
        prompt=prompt,
        kwargs=[kwargs],
    )


def _checker():
    return instructions_registry.INSTRUCTION_DICT[LETTER_FREQUENCY](LETTER_FREQUENCY)


@pytest.mark.parametrize(
    "key, prompt, kwargs, following, not_following", PUNCTUATION_PROMPTS
)
def test_punctuation_target_is_scored_deterministically(
    key, prompt, kwargs, following, not_following
):
    inp = _make_input(key, prompt, kwargs)
    for seed in range(20):
        random.seed(seed)
        assert ifeval_utils.test_instruction_following_strict(
            inp, following
        ).follow_all_instructions
        random.seed(seed)
        assert not ifeval_utils.test_instruction_following_strict(
            inp, not_following
        ).follow_all_instructions
        random.seed(seed)
        assert ifeval_utils.test_instruction_following_loose(
            inp, following
        ).follow_all_instructions


@pytest.mark.parametrize("letter", ["#", "!", "A", " e "])
def test_given_character_is_kept(letter):
    checker = _checker()
    checker.build_description(letter=letter, let_frequency=2, let_relation="at least")
    assert checker.get_instruction_args()["letter"] == letter.strip().lower()


def test_missing_letter_still_draws_a_letter():
    checker = _checker()
    checker.build_description(let_frequency=2, let_relation="at least")
    assert checker.get_instruction_args()["letter"] in string.ascii_lowercase


@pytest.mark.parametrize("letter", ["ab", "", "   "])
def test_malformed_letter_is_rejected(letter):
    with pytest.raises(ValueError, match="single non-whitespace character"):
        _checker().build_description(
            letter=letter, let_frequency=2, let_relation="at least"
        )
