import logging
from typing import List, Mapping, Tuple, Type, Optional, Union
from promptsource.templates import DatasetTemplates

import lm_eval.api.utils
from lm_eval.api.task import Task

from . import anli
from . import blimp
from . import diabla
from . import cnn_dailymail
from . import coqa
from . import crd3
from . import crows_pairs_multilingual
from . import drop
from . import e2e_nlg_cleaned
from . import flores_101
from . import gem_asset_turk
from . import gem_mlsum
from . import gem_webnlg
from . import gem_wikilingua
from . import gem_xsum
from . import glue
from . import hans
from . import huff_post
from . import jigsaw_unintended_bias
from . import lama
from . import lince
from . import piaf
from . import race
from . import schema_guided_dstc8
from . import superglue
from . import tydiqa
from . import wino_bias
from . import wmt
from . import xquad


logger = logging.getLogger(__name__)


TASK_REGISTRY = {
    # GLUE
    "cola": glue.CoLA,
    "mnli": glue.MNLI,
    "mnli_mismatched": glue.MNLIMismatched,
    "mrpc": glue.MRPC,
    "rte": glue.RTE,
    "qnli": glue.QNLI,
    "qqp": glue.QQP,
    # "stsb": glue.STSB, # not implemented yet
    "sst": glue.SST,
    "wnli": glue.WNLI,
    # SuperGLUE
    "axb": superglue.BroadcoverageDiagnostics,
    "axg": superglue.WinogenderSchemaDiagnostics,
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "record": superglue.ReCoRD,
    "superglue_rte": superglue.RTE,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    # CoQA
    "coqa": coqa.CoQA,
    # DROP
    "drop": drop.DROP,
    # E2E NLG
    "e2e_nlg_cleaned": e2e_nlg_cleaned.E2E_NLG_Cleaned,
    # DSTC8
    "schema_guided_dstc8": schema_guided_dstc8.Schema_Guided_DSTC8,
    # RACE
    "race": race.RACE,
    # ANLI
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    # HANS
    "hans": hans.HANS,
    # CNN Daily Mail
    "cnn_dailymail": cnn_dailymail.CnnDailyMail,
    # GEM/xum
    "gem_xsum": gem_xsum.GEMXSUM,
    "gem_xsum_challenge_sample": gem_xsum.GEMXSUMChallgeSample,
    "gem_xsum_challenge_test_backtranslation": gem_xsum.GEMXSUMChallgeTestBacktranslation,
    "gem_xsum_challenge_test_bfp_02": gem_xsum.GEMXSUMChallgeTestBFP02,
    "gem_xsum_challenge_test_bfp_05": gem_xsum.GEMXSUMChallgeTestBFP05,
    "gem_xsum_challenge_test_nopunc": gem_xsum.GEMXSUMChallgeTestNopunc,
    "gem_xsum_challenge_test_covid": gem_xsum.GEMXSUMChallgeTestCovid,
    # LAMA
    "lama-trex": lama.Trex,
    "lama-squad": lama.Squad,
    "lama-google_re": lama.google_re,
    "lama-conceptnet": lama.Conceptnet,
    # WinoBias
    "wino_bias_type1_pro": wino_bias.WinoBiasType1Pro,
    "wino_bias_type1_anti": wino_bias.WinoBiasType1Anti,
    "wino_bias_type2_pro": wino_bias.WinoBiasType2Pro,
    "wino_bias_type2_anti": wino_bias.WinoBiasType2Anti,
    # Crows-Pairs
    "crows_pairs_english": crows_pairs_multilingual.CrowsPairsEnglish,
    "crows_pairs_french": crows_pairs_multilingual.CrowsPairsFrench,
    # News
    "huffpost": huff_post.HuffPost,
    # Code-switching
    "lince_sa": lince.LinCESentimentAnalysis,
    # CRD3
    "crd3": crd3.CRD3,
    # DiaBLa
    "diabla": diabla.DiaBLa,
    "diabla_1_shot_context": diabla.DiaBLa_1_shot_context,
    # XQuAD
    "xquad_en": xquad.XQuADEnglish,
    "xquad_ar": xquad.XQuADArabic,
    # PIAF
    "piaf": piaf.PIAF,
    # Flores 101 (MT)
    "flores_101_mt": flores_101.Flores101MT,
    "flores_101_mt_fewshot_fr2en": flores_101.Flores101MT_fewshot_fr2en,
    "flores_101_mt_fewshot_hi2en": flores_101.Flores101MT_fewshot_hi2en,
    "flores_101_mt_fewshot_fr2ar": flores_101.Flores101MT_fewshot_fr2ar,
    "flores_101_mt_fewshot_wmt_fr2en": flores_101.Flores101MT_fewshot_wmt_fr2en,
    # Flores101 (Perplexity)
    "flores_101_ppl": flores_101.Flores101Perplexity,
    # GEM/WebNLG
    # Format: `GEM/web_nlg_{webnlg.subset_name}_{split}`
    **gem_webnlg.construct_tasks(),
    # GEM/WikiAssetTurk
    # Format: `GEM/wiki_auto_asset_turk_{split}`
    **gem_asset_turk.construct_tasks(),
    # GEM WikiLingua
    # Format: `GEM/wiki_lingua_{lang}`
    **gem_wikilingua.construct_tasks(),
    # WMT
    # Format: `wmt{year}_{lang1}_{lang2}`
    **wmt.construct_tasks(),
    # BLiMP
    "blimp_adjunct_island": blimp.BlimpAdjunctIsland,
    "blimp_anaphor_gender_agreement": blimp.BlimpAnaphorGenderAgreement,
    "blimp_anaphor_number_agreement": blimp.BlimpAnaphorNumberAgreement,
    "blimp_animate_subject_passive": blimp.BlimpAnimateSubjectPassive,
    "blimp_animate_subject_trans": blimp.BlimpAnimateSubjectTrans,
    "blimp_causative": blimp.BlimpCausative,
    "blimp_complex_NP_island": blimp.BlimpComplex_NPIsland,
    "blimp_coordinate_structure_constraint_complex_left_branch": blimp.BlimpCoordinateStructureConstraintComplexLeftBranch,
    "blimp_coordinate_structure_constraint_object_extraction": blimp.BlimpCoordinateStructureConstraintObjectExtraction,
    "blimp_determiner_noun_agreement_1": blimp.BlimpDeterminerNounAgreement_1,
    "blimp_determiner_noun_agreement_2": blimp.BlimpDeterminerNounAgreement_2,
    "blimp_determiner_noun_agreement_irregular_1": blimp.BlimpDeterminerNounAgreementIrregular_1,
    "blimp_determiner_noun_agreement_irregular_2": blimp.BlimpDeterminerNounAgreementIrregular_2,
    "blimp_determiner_noun_agreement_with_adj_2": blimp.BlimpDeterminerNounAgreementWithAdj_2,
    "blimp_determiner_noun_agreement_with_adj_irregular_1": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_1,
    "blimp_determiner_noun_agreement_with_adj_irregular_2": blimp.BlimpDeterminerNounAgreementWithAdjIrregular_2,
    "blimp_determiner_noun_agreement_with_adjective_1": blimp.BlimpDeterminerNounAgreementWithAdjective_1,
    "blimp_distractor_agreement_relational_noun": blimp.BlimpDistractorAgreementRelationalNoun,
    "blimp_distractor_agreement_relative_clause": blimp.BlimpDistractorAgreementRelativeClause,
    "blimp_drop_argument": blimp.BlimpDropArgument,
    "blimp_ellipsis_n_bar_1": blimp.BlimpEllipsisNBar_1,
    "blimp_ellipsis_n_bar_2": blimp.BlimpEllipsisNBar_2,
    "blimp_existential_there_object_raising": blimp.BlimpExistentialThereObjectRaising,
    "blimp_existential_there_quantifiers_1": blimp.BlimpExistentialThereQuantifiers_1,
    "blimp_existential_there_quantifiers_2": blimp.BlimpExistentialThereQuantifiers_2,
    "blimp_existential_there_subject_raising": blimp.BlimpExistentialThereSubjectRaising,
    "blimp_expletive_it_object_raising": blimp.BlimpExpletiveItObjectRaising,
    "blimp_inchoative": blimp.BlimpInchoative,
    "blimp_intransitive": blimp.BlimpIntransitive,
    "blimp_irregular_past_participle_adjectives": blimp.BlimpIrregularPastParticipleAdjectives,
    "blimp_irregular_past_participle_verbs": blimp.BlimpIrregularPastParticipleVerbs,
    "blimp_irregular_plural_subject_verb_agreement_1": blimp.BlimpIrregularPluralSubjectVerbAgreement_1,
    "blimp_irregular_plural_subject_verb_agreement_2": blimp.BlimpIrregularPluralSubjectVerbAgreement_2,
    "blimp_left_branch_island_echo_question": blimp.BlimpLeftBranchIslandEchoQuestion,
    "blimp_left_branch_island_simple_question": blimp.BlimpLeftBranchIslandSimpleQuestion,
    "blimp_matrix_question_npi_licensor_present": blimp.BlimpMatrixQuestionNpiLicensorPresent,
    "blimp_npi_present_1": blimp.BlimpNpiPresent_1,
    "blimp_npi_present_2": blimp.BlimpNpiPresent_2,
    "blimp_only_npi_licensor_present": blimp.BlimpOnlyNpiLicensorPresent,
    "blimp_only_npi_scope": blimp.BlimpOnlyNpiScope,
    "blimp_passive_1": blimp.BlimpPassive_1,
    "blimp_passive_2": blimp.BlimpPassive_2,
    "blimp_principle_A_c_command": blimp.BlimpPrinciple_ACCommand,
    "blimp_principle_A_case_1": blimp.BlimpPrinciple_ACase_1,
    "blimp_principle_A_case_2": blimp.BlimpPrinciple_ACase_2,
    "blimp_principle_A_domain_1": blimp.BlimpPrinciple_ADomain_1,
    "blimp_principle_A_domain_2": blimp.BlimpPrinciple_ADomain_2,
    "blimp_principle_A_domain_3": blimp.BlimpPrinciple_ADomain_3,
    "blimp_principle_A_reconstruction": blimp.BlimpPrinciple_AReconstruction,
    "blimp_regular_plural_subject_verb_agreement_1": blimp.BlimpRegularPluralSubjectVerbAgreement_1,
    "blimp_regular_plural_subject_verb_agreement_2": blimp.BlimpRegularPluralSubjectVerbAgreement_2,
    "blimp_sentential_negation_npi_licensor_present": blimp.BlimpSententialNegationNpiLicensorPresent,
    "blimp_sentential_negation_npi_scope": blimp.BlimpSententialNegationNpiScope,
    "blimp_sentential_subject_island": blimp.BlimpSententialSubjectIsland,
    "blimp_superlative_quantifiers_1": blimp.BlimpSuperlativeQuantifiers_1,
    "blimp_superlative_quantifiers_2": blimp.BlimpSuperlativeQuantifiers_2,
    "blimp_tough_vs_raising_1": blimp.BlimpToughVsRaising_1,
    "blimp_tough_vs_raising_2": blimp.BlimpToughVsRaising_2,
    "blimp_transitive": blimp.BlimpTransitive,
    "blimp_wh_island": blimp.BlimpWhIsland,
    "blimp_wh_questions_object_gap": blimp.BlimpWhQuestionsObjectGap,
    "blimp_wh_questions_subject_gap": blimp.BlimpWhQuestionsSubjectGap,
    "blimp_wh_questions_subject_gap_long_distance": blimp.BlimpWhQuestionsSubjectGapLongDistance,
    "blimp_wh_vs_that_no_gap": blimp.BlimpWhVsThatNoGap,
    "blimp_wh_vs_that_no_gap_long_distance": blimp.BlimpWhVsThatNoGapLongDistance,
    "blimp_wh_vs_that_with_gap": blimp.BlimpWhVsThatWithGap,
    "blimp_wh_vs_that_with_gap_long_distance": blimp.BlimpWhVsThatWithGapLongDistance,
    # TyDi QA
    "tydiqa_primary": tydiqa.TyDiQAPrimaryClassification,
    "tydiqa_secondary": tydiqa.TyDiQAGoldPGeneration,
    #######################################################
    # TODO: Not Yet Available in `promptsource/eval-hackathon`
    ########################################################
    # GEM/mlsum
    # "mlsum_es": gem_mlsum.GEMMLSUMEs,
    # "mlsum_de": gem_mlsum.GEMMLSUMDe,
    # "mlsum_es_covid_challenge_set": gem_mlsum.GEMMLSUMEsChallgeTestCovid,
    # "mlsum_de_covid_challenge_set": gem_mlsum.GEMMLSUMDeChallgeTestCovid,
    # LAMA
    # "bigscience-lama": lama.BigScienceLAMA,
    ########################################################
    # TODO: Tasks That Require Manual Download:
    ########################################################
    # JigSaw
    # "jigsaw_unintended_bias": jigsaw_unintended_bias.JigsawUnintendedBias,
    ########################################################
}


def list_tasks() -> List[str]:
    """Returns a list of all the available tasks by name."""
    return sorted(list(TASK_REGISTRY))


def get_task(task_name: str, template_name: str, **task_kwargs) -> Task:
    """Returns a task from the registry and instantiates it with the `promptsource`
    template specified by `template_name`.

    Args:
        task_name: Name of the task to load from the task registry.
        template_name: Name of the prompt template from `promptsource` to use
            for this task.
        **task_kwargs: Keyword arguments to pass to the task constructor. See constructor
            args for `lm_eval.api.task.Task`.

    Returns:
        A task instance with formatting specified by `template_name`.
    """
    task_class = _get_task_from_registry(task_name)
    template = get_templates(task_name)[template_name]
    return task_class(prompt_template=template, **task_kwargs)


def get_task_list(
    task_name: str, template_names: List[str], **task_kwargs
) -> List[Task]:
    """Returns a list of the same task but with multiple prompt templates.

    Args:
        task_name: Name of the task to load from the task registry.
        template_names: Name of the prompt template from `promptsource` to use
            for this task.
        **task_kwargs: Keyword arguments to pass to the task constructor. See constructor
            args for `lm_eval.api.task.Task`.

    Returns:
        A list of tasks with the same name but different prompt templates.
    """
    assert template_names, "Must specify at least one template name"
    template_names = sorted(set(template_names))
    return [get_task(task_name, t, **task_kwargs) for t in template_names]


def list_templates(task_name: str) -> List[str]:
    """Returns all template names available in `promptsource` for a given task."""
    templates = get_templates(task_name)
    return sorted(templates.all_template_names)


def get_templates(task_name: str) -> DatasetTemplates:
    """Returns the `promptsource` `DatasetTemplates` for the specified task name."""
    task_class = _get_task_from_registry(task_name)
    return _get_templates_from_task(task_class)


def get_task_list_from_args_string(
    task_name: str,
    template_names: List[str],
    task_args: str,
    additional_config: Optional[Mapping[str, str]] = None,
) -> List[Task]:
    """Returns a list of the same task but with multiple prompt templates, each
    task instantiated with the given kwargs.

    Args:
        task_name: Name of the task to use as found in the task registry.
        template_names: Name of the prompt template from `promptsource` to use
            for this task.
        task_args: A string of comma-separated key=value pairs that will be passed
            to the task constructor. E.g. "data_dir=./datasets,example_separator=\n\n"
        additional_config: An additional dictionary of key=value pairs that will
            be passed to the task constructor.

    Returns:
        A list of `Task` instances.
    """
    kwargs = lm_eval.api.utils.parse_cli_args_string(task_args)
    assert "prompt_template" not in kwargs, (
        "Cannot specify a `prompt_template` object in the `task_args` string. "
        "Only primitive type arguments are allowed."
    )
    additional_config = {} if additional_config is None else additional_config
    additional_args = {k: v for k, v in additional_config.items() if v is not None}
    kwargs.update(additional_args)
    return get_task_list(task_name, template_names, **kwargs)


# Helper functions


def _get_task_from_registry(task_name: str) -> Type[Task]:
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        logger.warning(f"Available tasks:\n{list_tasks()}")
        raise KeyError(f"`{task_name}` is missing from the task registry.")


def _get_templates_from_task(task: Union[Task, Type[Task]]) -> DatasetTemplates:
    dataset_name = (
        task.DATASET_PATH
        if task.DATASET_NAME is None
        else f"{task.DATASET_PATH}/{task.DATASET_NAME}"
    )
    return DatasetTemplates(dataset_name)


# TODO(jon-tow): Refactor everything below! These functions are only required
# b/c the task registry is non-uniformly hard-coded.


# TODO(jon-tow): Remove this function after refactoring the task registry to use
# `Task` object __str__ representations for task names as opposed to
# hardcoded string keys.
def get_registry_name_from_task(task: Task) -> str:
    """Returns the task registry name from a Task instance."""
    for name, class_ in TASK_REGISTRY.items():
        if isinstance(task, class_):
            return name
    # This gives a mechanism for non-registered tasks to have a custom name anyways when reporting.
    return type(task).__name__


_TASK_TEMPLATE_KEY_SEP = "+"


def _get_task_template_key(task_name: str, template_name: str) -> str:
    """Returns a `str` key for a task with that prompt template name appended.
    This should be used to uniquely identify a task by its name AND
    its specific prompt template - as a task can have many templates.
    """
    if not template_name:
        # Add `null` prompt template to the key if no template name is specified.
        template_name = "null"
    return f"{task_name}{_TASK_TEMPLATE_KEY_SEP}{template_name}"


def _split_task_template_key(task_template_key: str) -> Tuple[str, str]:
    """Splits a task template key as returned from `_get_task_template_key`
    into it's constituent parts: (task name, template name).
    """
    task_name, template_name = task_template_key.split(_TASK_TEMPLATE_KEY_SEP, 1)
    return task_name, template_name
