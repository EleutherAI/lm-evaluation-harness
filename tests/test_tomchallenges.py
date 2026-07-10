from lm_eval.tasks.tomchallenges import utils as tomchallenges_utils


def test_load_counts_and_story_families():
    dataset = tomchallenges_utils.load()['test']
    assert len(dataset) == 360

    story_families = [row['story_family'] for row in dataset]
    assert story_families.count('sally_anne') == 180
    assert story_families.count('smarties') == 180



def test_belief_group_counts():
    dataset = tomchallenges_utils.load()['test']
    assert len(tomchallenges_utils.process_docs_reality(dataset)) == 60
    assert len(tomchallenges_utils.process_docs_anti_reality(dataset)) == 60
    assert len(tomchallenges_utils.process_docs_first_order_a(dataset)) == 60
    assert len(tomchallenges_utils.process_docs_first_order_b(dataset)) == 60
    assert len(tomchallenges_utils.process_docs_second_order_a(dataset)) == 60
    assert len(tomchallenges_utils.process_docs_second_order_b(dataset)) == 60


def test_reconstructs_gold_mc_and_tf_labels():
    dataset = tomchallenges_utils.load()['test']
    sally = next(
        row for row in dataset if row['story_family'] == 'sally_anne' and row['question_type'] == 'reality'
    )
    smarties = next(
        row for row in dataset if row['story_family'] == 'smarties' and row['question_type'] == 'assumption'
    )

    assert sally['mc_choices'] == ['cabinet', 'closet']
    assert sally['mc_gold_index'] == 0
    assert sally['tf_gold_a'] is True
    assert sally['tf_gold_b'] is False

    assert smarties['mc_choices'] == ['vest', 'plate']
    assert smarties['mc_gold_index'] == 1
    assert smarties['tf_gold_a'] is False
    assert smarties['tf_gold_b'] is True


def test_open_and_tf_scoring_helpers():
    dataset = tomchallenges_utils.load()['test']
    sally = next(
        row for row in dataset if row['story_family'] == 'sally_anne' and row['question_type'] == 'reality'
    )
    smarties = next(
        row for row in dataset if row['story_family'] == 'smarties' and row['question_type'] == 'assumption'
    )

    assert tomchallenges_utils.process_results_open(
        sally, ['The towel is currently in the cabinet.']
    ) == {'acc': 1.0}
    assert tomchallenges_utils.process_results_open(
        sally, ['The towel is currently in the closet.']
    ) == {'acc': 0.0}
    assert tomchallenges_utils.process_results_mc(sally, ['A']) == {'acc': 1.0}
    assert tomchallenges_utils.process_results_mc(sally, ['B']) == {'acc': 0.0}
    assert tomchallenges_utils.process_results_tf(
        smarties, ['A. false\nB. true']
    ) == {'acc': 1.0}
    assert tomchallenges_utils.process_results_tf(
        smarties, ['A. true\nB. false']
    ) == {'acc': 0.0}


def test_tfr_prefers_leading_judgment_token():
    dataset = tomchallenges_utils.load()['test']
    smarties = next(
        row for row in dataset if row['story_family'] == 'smarties' and row['question_type'] == '1stA'
    )
    response = (
        'A. True. Sabra already found a pepper in the backpack and there is no indication '
        'that anything else was added or removed from the backpack before Hillary opened it.\n'
        'B. False. There is no indication that a fork was ever in the backpack, so Sabra '
        'would not expect to find one after Hillary opened it.'
    )
    assert tomchallenges_utils._extract_tf_pair(response) == (True, False)
    assert tomchallenges_utils.process_results_tf(smarties, [response]) == {'acc': 1.0}
