from promptsource.templates import DatasetTemplates
from pprint import pprint
from typing import List, Union

import lm_eval.base

from . import anli
from . import blimp
from . import coqa
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
from . import lama
from . import lince
from . import race
from . import scitail
from . import superglue
from . import wino_bias
from . import wmt
from . import crd3
from . import cnn_dailymail
from . import diabla
from . import xquad
from . import schema_guided_dstc8
from . import piaf


########################################
# All tasks
########################################


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
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "multirc": superglue.MultiRC,
    "record": superglue.ReCoRD,
    "wic": superglue.WordsInContext,
    "wsc": superglue.SGWinogradSchemaChallenge,
    "axg": superglue.WinogenderSchemaDiagnostics,
    "axb": superglue.BroadcoverageDiagnostics,
    # Order by benchmark/genre?
    "coqa": coqa.CoQA,
    "drop": drop.DROP,
    **gem_webnlg.construct_tasks(),
    # multilingual lambada
    **gem_asset_turk.construct_tasks(),
    # GEM WikiLingua
    **gem_wikilingua.construct_tasks(),
    "e2e_nlg_cleaned": e2e_nlg_cleaned.E2E_NLG_Cleaned,
    "schema_guided_dstc8": schema_guided_dstc8.Schema_Guided_DSTC8,
    # formatted as gsarti/flores_101_[LANG]
    **flores_101.construct_tasks(),
    "lama_trex": lama.Trex,
    "lama_squad": lama.Squad,
    "lama_google_re": lama.google_re,
    "lama_concptnet": lama.Conceptnet,
    "race": race.RACE,
    # "naturalqs": naturalqs.NaturalQs, # not implemented yet
    "anli_r1": anli.ANLIRound1,
    "anli_r2": anli.ANLIRound2,
    "anli_r3": anli.ANLIRound3,
    "hans": hans.HANS,
    
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
    
    # CNN Daily Mail
    "cnn_dailymail": cnn_dailymail.CnnDailyMail,
    
    # GEM/mlsum
    "mlsum_es": gem_mlsum.GEMMLSUMEs,
    "mlsum_de": gem_mlsum.GEMMLSUMDe,
    "mlsum_es_covid_challenge_set": gem_mlsum.GEMMLSUMEsChallgeTestCovid,
    "mlsum_de_covid_challenge_set": gem_mlsum.GEMMLSUMDeChallgeTestCovid,
    
    # Requires manual download of data.
    # "storycloze_2016": storycloze.StoryCloze2016,
    # "storycloze_2018": storycloze.StoryCloze2018,
    # "sat": sat.SATAnalogies,
    
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
    "lama-concptnet": lama.Conceptnet,
    "bigscience-lama": lama.BigScienceLAMA,
    
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
    # WMT
    **wmt.create_year_tasks(wmt.WMT14_TASKS),
    
    # DiaBLa
    "diabla": diabla.DiaBLa,
    # XQuAD
    "xquad_en": xquad.XQuADEnglish,
    "xquad_ar": xquad.XQuADArabic,

    # piaf
    "piaf": piaf.PIAF,
    
    # SciTail
    "scitail": scitail.SciTailTE,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as e:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, lm_eval.base.Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}


def get_task_dict_promptsource(task_name_list: List[str]):
    """Loads a task instance for each prompt written for that task."""
    task_name_dict = {}

    for task_name in task_name_list:
        assert isinstance(task_name, str)

        # Static version of the Task Use this to get HF dataset path / name.
        static_task_obj = get_task(task_name)
        if issubclass(static_task_obj, lm_eval.base.BioTask):
            # Create the proper task name arg for DatasetTemplates.
            sub_task = (
                f"/{static_task_obj.DATASET_NAME}"
                if static_task_obj.DATASET_NAME
                else ""
            )
            ps_task_name = static_task_obj.DATASET_PATH.split("/")[-1] + "/"
            ps_task_name += static_task_obj.DATASET_NAME

            task_prompts = DatasetTemplates(ps_task_name)
            for prompt_name in task_prompts.all_template_names:
                prompt = task_prompts[prompt_name]
                # NOTE: We choose a sep that can be easily split.
                task_name_dict[f"{task_name}+{prompt_name}"] = get_task(task_name)(
                    prompt=prompt
                )
        elif issubclass(static_task_obj, lm_eval.base.PromptSourceTask):
            # Create the proper task name arg for DatasetTemplates.
            sub_task = (
                f"/{static_task_obj.DATASET_NAME}"
                if static_task_obj.DATASET_NAME
                else ""
            )
            ps_task_name = f"{static_task_obj.DATASET_PATH}{sub_task}"

            task_prompts = DatasetTemplates(ps_task_name)
            for prompt_name in task_prompts.all_template_names:
                prompt = task_prompts[prompt_name]
                # NOTE: We choose a sep that can be easily split.
                task_name_dict[f"{task_name}+{prompt_name}"] = get_task(task_name)(
                    prompt=prompt
                )
        
        else:
            # This is a task with a null prompt.
            # Right now, the only use case are `PerplexityTask`s.
            task_name_dict[f"{task_name}+null"] = static_task_obj()
    return task_name_dict
