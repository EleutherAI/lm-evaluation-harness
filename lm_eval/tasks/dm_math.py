"""
Analysing Mathematical Reasoning Abilities of Neural Models
https://arxiv.org/pdf/1904.01557.pdf

*Describe dataset here*

Homepage: https://github.com/deepmind/mathematics_dataset
"""
from lm_eval.metrics import mean
from lm_eval.base import Task, rf


_CITATION = """
ADD CITATION HERE
"""


class DMMath(Task):
    DATASET_PATH = "math_dataset"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # dataset fields are in format "b'{string contents here}'". 
        # we want to strip this extraneous bytes formatting from the strings.
        doc["answer"] = doc["answer"].lstrip("b'").rstrip("'")
        doc["question"] = doc["question"].lstrip("b'").rstrip("'")
        return doc

    def doc_to_text(self, doc):
        return doc["question"] + "Answer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        return " " + doc["answer"]

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, ["\n", "\n\n"])

    def process_results(self, doc, results):
        if doc["answer"].rstrip("\n") == results[0].rstrip("\n"): # for now, simple string comparison. TODO: sympy answer checking, especially for harder subsets that don't just return a number
            is_correct = 1
        else:
            is_correct = 0
        return {"acc": is_correct}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


class DMMathLinAlg1d(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__linear_1d"

class DMMathLinAlg1dComp(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__linear_1d_composed"

class DMMathLinAlg2d(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__linear_2d"

class DMMathLinAlg2dComp(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__linear_2d"

class DMMathPolyRoots(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__polynomial_roots"

class DMMathPolyRootsComp(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__polynomial_roots_composed"

class DMMathSeqNext(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__sequence_next_term"

class DMMathSeqNth(DMMath):
    VERSION = 0
    DATASET_NAME = "algebra__sequence_nth_term"

class DMMathAddOrSub(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__add_or_sub"

class DMMathAddOrSubBase(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__add_or_sub_in_base"

class DMMathAddOrSubComp(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__add_sub_multiple"

class DMMathDiv(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__div"

class DMMathMixed(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__mixed"

class DMMathMult(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__mul"

class DMMathMultDivComp(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__mul_div_multiple"

class DMMathNearestRoot(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__nearest_integer_root"

class DMMathSimplify(DMMath):
    VERSION = 0
    DATASET_NAME = "arithmetic__simplify_surd"

class DMMathDiff(DMMath):
    VERSION = 0
    DATASET_NAME = "calculus__differentiate"

class DMMathDiffComp(DMMath):
    VERSION = 0
    DATASET_NAME = "calculus__differentiate_composed"

class DMMathClosest(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__closest"

class DMMathClosestComp(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__closest_composed"

class DMMathKthBiggest(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__kth_biggest"

class DMMathKthBiggestComp(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__kth_biggest_composed"

class DMMathPair(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__pair"

class DMMathPairComp(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__pair_composed"

class DMMathSort(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__sort"

class DMMathSortComp(DMMath):
    VERSION = 0
    DATASET_NAME = "comparison__sort_composed"

class DMMathMeasConv(DMMath):
    VERSION = 0
    DATASET_NAME = "measurement__conversion"

class DMMathMeasTime(DMMath):
    VERSION = 0
    DATASET_NAME = "measurement__time"

class DMMathBaseConv(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__base_conversion"

class DMMathDivRemainder(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__div_remainder"

class DMMathDivRemainderComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__div_remainder_composed"

class DMMathGcd(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__gcd"

class DMMathGcdComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__gcd_composed"

class DMMathIsFactor(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__is_factor"

class DMMathIsFactorComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__is_factor_composed"

class DMMathLcm(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__lcm"

class DMMathLcmComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__lcm_composed"

class DMMathListPrimeFactors(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__list_prime_factors"

class DMMathListPrimeFactorsComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__list_prime_factors_composed"

class DMMathPlaceVal(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__place_value"

class DMMathPlaceValComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__place_value_composed"

class DMMathRoundNum(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__round_number"

class DMMathRoundNumComp(DMMath):
    VERSION = 0
    DATASET_NAME = "numbers__round_number_composed"

class DMMathAddPoly(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__add"

class DMMathPolyCoeff(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__coefficient_named"

class DMMathPolyCollect(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__collect"

class DMMathPolyComp(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__compose"

class DMMathPolyEval(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__evaluate"

class DMMathPolyEvalComp(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__evaluate_composed"

class DMMathPolyExpand(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__expand"

class DMMathPolySimplify(DMMath):
    VERSION = 0
    DATASET_NAME = "polynomials__simplify_power"

class DMMathProbLevelSet(DMMath):
    VERSION = 0
    DATASET_NAME = "probability__swr_p_level_set"

class DMMathProbSeq(DMMath):
    VERSION = 0
    DATASET_NAME = "probability__swr_p_sequence"