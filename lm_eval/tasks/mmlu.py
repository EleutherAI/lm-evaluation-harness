"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/abs/2009.03300

We propose a new test to measure a text model's multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history,
computer science, law, and more. To attain high accuracy on this test,
models must possess extensive world knowledge and problem solving ability.
We find that while most recent models have near random-chance accuracy,
the very largest GPT-3 model improves over random chance by almost 20
percentage points on average. However, on every one of the 57 tasks,
the best models still need substantial improvements before they can
reach expert-level accuracy. Models also have lopsided performance and
frequently do not know when they are wrong. Worse, they still have
near-random accuracy on some socially important subjects such as
morality and law. By comprehensively evaluating the breadth and
depth of a model's academic and professional understanding, our
test can be used to analyze models across many tasks and to
identify important shortcomings.

Homepage: https://paperswithcode.com/dataset/mmlu
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{DBLP:journals/corr/abs-2009-03300,
  author       = {Dan Hendrycks and
                  Collin Burns and
                  Steven Basart and
                  Andy Zou and
                  Mantas Mazeika and
                  Dawn Song and
                  Jacob Steinhardt},
  title        = {Measuring Massive Multitask Language Understanding},
  journal      = {CoRR},
  volume       = {abs/2009.03300},
  year         = {2020},
  url          = {https://arxiv.org/abs/2009.03300},
  eprinttype    = {arXiv},
  eprint       = {2009.03300},
  timestamp    = {Thu, 17 Sep 2020 12:49:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2009-03300.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""


class MMLUBase(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "lukaemon/mmlu"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        choices = [doc["A"], doc["B"], doc["C"], doc["D"]]
        out_doc = {
            **doc,
            "choices": choices,
            "gold": ["A", "B", "C", "D"].index(doc["target"]),
        }
        return out_doc

    @staticmethod
    def format_choices(choices):
        result = ""
        for choice in choices:
            result += f"- {choice}\n"

        return result

    def doc_to_text(self, doc):
        input = doc["input"]
        options = MMLUBase.format_choices(doc["choices"])
        prompt = f"""Question: {input}
Options:
{options}
Answer:"""

        return prompt


# Full list scraped from huggingface website
"""
abstract_algebra
anatomy
astronomy
business_ethics
clinical_knowledge
college_biology
college_chemistry
college_computer_science
college_mathematics
college_medicine
college_physics
computer_security
conceptual_physics
econometrics
electrical_engineering
elementary_mathematics
formal_logic
global_facts
high_school_biology
high_school_chemistry
high_school_computer_science
high_school_european_history
high_school_geography
high_school_government_and_politics
high_school_macroeconomics
high_school_mathematics
high_school_microeconomics
high_school_physics
high_school_psychology
high_school_statistics
high_school_us_history
high_school_world_history
human_aging
human_sexuality
international_law
jurisprudence
logical_fallacies
machine_learning
management
marketing
medical_genetics
miscellaneous
moral_disputes
moral_scenarios
nutrition
philosophy
prehistory
professional_accounting
professional_law
professional_medicine
professional_psychology
public_relations
security_studies
sociology
us_foreign_policy
virology
world_religions
"""


# Thank you ChatGPT!!!
class MMLUAbstractAlgebra(MMLUBase):
    DATASET_NAME = "abstract_algebra"


class MMLUAnatomy(MMLUBase):
    DATASET_NAME = "anatomy"


class MMLUAstronomy(MMLUBase):
    DATASET_NAME = "astronomy"


class MMLUBusinessEthics(MMLUBase):
    DATASET_NAME = "business_ethics"


class MMLUClinicalKnowledge(MMLUBase):
    DATASET_NAME = "clinical_knowledge"


class MMLUCollegeBiology(MMLUBase):
    DATASET_NAME = "college_biology"


class MMLUCollegeChemistry(MMLUBase):
    DATASET_NAME = "college_chemistry"


class MMLUCollegeComputerScience(MMLUBase):
    DATASET_NAME = "college_computer_science"


class MMLUCollegeMathematics(MMLUBase):
    DATASET_NAME = "college_mathematics"


class MMLUCollegeMedicine(MMLUBase):
    DATASET_NAME = "college_medicine"


class MMLUCollegePhysics(MMLUBase):
    DATASET_NAME = "college_physics"


class MMLUComputerSecurity(MMLUBase):
    DATASET_NAME = "computer_security"


class MMLUConceptualPhysics(MMLUBase):
    DATASET_NAME = "conceptual_physics"


class MMLUEconometrics(MMLUBase):
    DATASET_NAME = "econometrics"


class MMLUElectricalEngineering(MMLUBase):
    DATASET_NAME = "electrical_engineering"


class MMLUElementaryMathematics(MMLUBase):
    DATASET_NAME = "elementary_mathematics"


class MMLUFormalLogic(MMLUBase):
    DATASET_NAME = "formal_logic"


class MMLUGlobalFacts(MMLUBase):
    DATASET_NAME = "global_facts"


class MMLUHighSchoolBiology(MMLUBase):
    DATASET_NAME = "high_school_biology"


class MMLUHighSchoolChemistry(MMLUBase):
    DATASET_NAME = "high_school_chemistry"


class MMLUHighSchoolComputerScience(MMLUBase):
    DATASET_NAME = "high_school_computer_science"


class MMLUHighSchoolEuropeanHistory(MMLUBase):
    DATASET_NAME = "high_school_european_history"


class MMLUHighSchoolGeography(MMLUBase):
    DATASET_NAME = "high_school_geography"


class MMLUHighSchoolGovernmentAndPolitics(MMLUBase):
    DATASET_NAME = "high_school_government_and_politics"


class MMLUHighSchoolMacroeconomics(MMLUBase):
    DATASET_NAME = "high_school_macroeconomics"


class MMLUHighSchoolMathematics(MMLUBase):
    DATASET_NAME = "high_school_mathematics"


class MMLUHighSchoolMicroeconomics(MMLUBase):
    DATASET_NAME = "high_school_microeconomics"


class MMLUHighSchoolPhysics(MMLUBase):
    DATASET_NAME = "high_school_physics"


class MMLUHighSchoolPsychology(MMLUBase):
    DATASET_NAME = "high_school_psychology"


class MMLUHighSchoolStatistics(MMLUBase):
    DATASET_NAME = "high_school_statistics"


class MMLUHighSchoolUSHistory(MMLUBase):
    DATASET_NAME = "high_school_us_history"


class MMLUHighSchoolWorldHistory(MMLUBase):
    DATASET_NAME = "high_school_world_history"


class MMLUHumanAging(MMLUBase):
    DATASET_NAME = "human_aging"


class MMLUHumanSexuality(MMLUBase):
    DATASET_NAME = "human_sexuality"


class MMLUInternationalLaw(MMLUBase):
    DATASET_NAME = "international_law"


class MMLUJurisprudence(MMLUBase):
    DATASET_NAME = "jurisprudence"


class MMLULogicalFallacies(MMLUBase):
    DATASET_NAME = "logical_fallacies"


class MMLUMachineLearning(MMLUBase):
    DATASET_NAME = "machine_learning"


class MMLUManagement(MMLUBase):
    DATASET_NAME = "management"


class MMLUMarketing(MMLUBase):
    DATASET_NAME = "marketing"


class MMLUMedicalGenetics(MMLUBase):
    DATASET_NAME = "medical_genetics"


class MMLUMiscellaneous(MMLUBase):
    DATASET_NAME = "miscellaneous"


class MMLUMoralDisputes(MMLUBase):
    DATASET_NAME = "moral_disputes"


class MMLUMoralScenarios(MMLUBase):
    DATASET_NAME = "moral_scenarios"


class MMLUNutrition(MMLUBase):
    DATASET_NAME = "nutrition"


class MMLUPhilosophy(MMLUBase):
    DATASET_NAME = "philosophy"


class MMLUPrehistory(MMLUBase):
    DATASET_NAME = "prehistory"


class MMLUProfessionalAccounting(MMLUBase):
    DATASET_NAME = "professional_accounting"


class MMLUProfessionalLaw(MMLUBase):
    DATASET_NAME = "professional_law"


class MMLUProfessionalMedicine(MMLUBase):
    DATASET_NAME = "professional_medicine"


class MMLUProfessionalPsychology(MMLUBase):
    DATASET_NAME = "professional_psychology"


class MMLUPublicRelations(MMLUBase):
    DATASET_NAME = "public_relations"


class MMLUSecurityStudies(MMLUBase):
    DATASET_NAME = "security_studies"


class MMLUSociology(MMLUBase):
    DATASET_NAME = "sociology"


class MMLUUSForeignPolicy(MMLUBase):
    DATASET_NAME = "us_foreign_policy"


class MMLUVirology(MMLUBase):
    DATASET_NAME = "virology"


class MMLUWorldReligions(MMLUBase):
    DATASET_NAME = "world_religions"
