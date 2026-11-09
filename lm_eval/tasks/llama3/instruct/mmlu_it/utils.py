from functools import partial

import datasets


def process_docs(dataset: datasets.Dataset, subtask) -> datasets.Dataset:
    return dataset.filter(
        lambda example: example["subtask_name"] == f"mmlu_it_chat.{subtask}"
    )


process_docs_miscellaneous = partial(process_docs, subtask="miscellaneous")
process_docs_high_school_physics = partial(process_docs, subtask="high_school_physics")
process_docs_high_school_computer_science = partial(
    process_docs, subtask="high_school_computer_science"
)
process_docs_high_school_statistics = partial(
    process_docs, subtask="high_school_statistics"
)
process_docs_professional_accounting = partial(
    process_docs, subtask="professional_accounting"
)
process_docs_machine_learning = partial(process_docs, subtask="machine_learning")
process_docs_econometrics = partial(process_docs, subtask="econometrics")
process_docs_astronomy = partial(process_docs, subtask="astronomy")
process_docs_business_ethics = partial(process_docs, subtask="business_ethics")
process_docs_high_school_macroeconomics = partial(
    process_docs, subtask="high_school_macroeconomics"
)
process_docs_jurisprudence = partial(process_docs, subtask="jurisprudence")
process_docs_professional_psychology = partial(
    process_docs, subtask="professional_psychology"
)
process_docs_high_school_chemistry = partial(
    process_docs, subtask="high_school_chemistry"
)
process_docs_philosophy = partial(process_docs, subtask="philosophy")
process_docs_college_medicine = partial(process_docs, subtask="college_medicine")
process_docs_medical_genetics = partial(process_docs, subtask="medical_genetics")
process_docs_high_school_microeconomics = partial(
    process_docs, subtask="high_school_microeconomics"
)
process_docs_high_school_geography = partial(
    process_docs, subtask="high_school_geography"
)
process_docs_college_biology = partial(process_docs, subtask="college_biology")
process_docs_human_aging = partial(process_docs, subtask="human_aging")
process_docs_anatomy = partial(process_docs, subtask="anatomy")
process_docs_logical_fallacies = partial(process_docs, subtask="logical_fallacies")
process_docs_clinical_knowledge = partial(process_docs, subtask="clinical_knowledge")
process_docs_conceptual_physics = partial(process_docs, subtask="conceptual_physics")
process_docs_human_sexuality = partial(process_docs, subtask="human_sexuality")
process_docs_formal_logic = partial(process_docs, subtask="formal_logic")
process_docs_abstract_algebra = partial(process_docs, subtask="abstract_algebra")
process_docs_high_school_biology = partial(process_docs, subtask="high_school_biology")
process_docs_marketing = partial(process_docs, subtask="marketing")
process_docs_world_religions = partial(process_docs, subtask="world_religions")
process_docs_high_school_european_history = partial(
    process_docs, subtask="high_school_european_history"
)
process_docs_college_computer_science = partial(
    process_docs, subtask="college_computer_science"
)
process_docs_high_school_world_history = partial(
    process_docs, subtask="high_school_world_history"
)
process_docs_prehistory = partial(process_docs, subtask="prehistory")
process_docs_high_school_mathematics = partial(
    process_docs, subtask="high_school_mathematics"
)
process_docs_global_facts = partial(process_docs, subtask="global_facts")
process_docs_moral_scenarios = partial(process_docs, subtask="moral_scenarios")
process_docs_electrical_engineering = partial(
    process_docs, subtask="electrical_engineering"
)
process_docs_management = partial(process_docs, subtask="management")
process_docs_elementary_mathematics = partial(
    process_docs, subtask="elementary_mathematics"
)
process_docs_us_foreign_policy = partial(process_docs, subtask="us_foreign_policy")
process_docs_professional_medicine = partial(
    process_docs, subtask="professional_medicine"
)
process_docs_college_physics = partial(process_docs, subtask="college_physics")
process_docs_high_school_government_and_politics = partial(
    process_docs, subtask="high_school_government_and_politics"
)
process_docs_security_studies = partial(process_docs, subtask="security_studies")
process_docs_professional_law = partial(process_docs, subtask="professional_law")
process_docs_high_school_us_history = partial(
    process_docs, subtask="high_school_us_history"
)
process_docs_virology = partial(process_docs, subtask="virology")
process_docs_nutrition = partial(process_docs, subtask="nutrition")
process_docs_college_chemistry = partial(process_docs, subtask="college_chemistry")
process_docs_computer_security = partial(process_docs, subtask="computer_security")
process_docs_public_relations = partial(process_docs, subtask="public_relations")
process_docs_moral_disputes = partial(process_docs, subtask="moral_disputes")
process_docs_college_mathematics = partial(process_docs, subtask="college_mathematics")
process_docs_high_school_psychology = partial(
    process_docs, subtask="high_school_psychology"
)
process_docs_international_law = partial(process_docs, subtask="international_law")
process_docs_sociology = partial(process_docs, subtask="sociology")
