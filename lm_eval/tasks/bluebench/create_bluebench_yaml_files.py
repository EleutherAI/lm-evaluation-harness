from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
import os

@dataclass
class _DefaultUnitxtRecipeArgs():
    card: str
    demos_pool_size: int = 100
    num_demos: List[int] = field(default_factory=lambda: [1])  # TODO: consider
    template: List[str] = field(default_factory=lambda: [])
    train_refiner: List[str] = field(default_factory=lambda: [])
    demos_taken_from: Literal["test", "validation", "train"] = "train"
    num_demos: List[int] = field(default_factory=lambda: [1])  # TODO: consider
    template_card_index: List[int] = field(default_factory=lambda: [1])  # TODO: consider
    system_prompt: List[str] = field(default_factory=lambda: ["system_prompts.empty"])
    format: List[str] = field(default_factory=lambda: ["formats.empty"])


subsets = {  # the key must appear in the card name
    "legalbench": [
        "abercrombie",
        "proa",
        "function_of_decision_section",
        "international_citizenship_questions",
        "corporate_lobbying",
    ],
    "mmlu_pro": [
        "history",
        "law",
        "health",
        "physics",
        "business",
        "other",
        "philosophy",
        "psychology",
        "economics",
        "math",
        "biology",
        "chemistry",
        "computer_science",
        "engineering",
    ],
    "bbq": [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_SES",
        "Race_x_gender",
        "Religion",
        "SES",
        "Sexual_orientation",
    ],
    "CFPB": ["watsonx", "2023"],
    "universal_ner": ["en.ewt"],  # , "en.pud"],

    "flores_101": [
        "ara_eng",
        "deu_eng",
        "eng_ara",
        "eng_deu",
        "eng_fra",
        "eng_kor",
        "eng_por",
        "eng_ron",
        "eng_spa",
        "fra_eng",
        "jpn_eng",
        "kor_eng",
        "por_eng",
        "ron_eng",
        "spa_eng",
    ],
}

unitxt_recipe_args_by_groupings: Dict[str, List[_DefaultUnitxtRecipeArgs]] = {
   "Reasoning": [
        _DefaultUnitxtRecipeArgs(
            card="cards.hellaswag",
            template=[
                    # "templates.completion.multiple_choice.simple",
                    # "templates.completion.multiple_choice.enumerated",
                    # "templates.completion.multiple_choice.standard"
                    "templates.completion.multiple_choice.blue_bench",
            ],
            # "templates.completion.multiple_choice.title",
            # ],
            num_demos=[5],
        ),
        _DefaultUnitxtRecipeArgs(
            card="cards.openbook_qa",
            template=[
                    # "templates.qa.multiple_choice.open.helm",
                    # "templates.qa.multiple_choice.open.lm_eval_harness",
                    # "templates.qa.multiple_choice.open.mmlu"
                    "templates.qa.multiple_choice.open.blue_bench",
            ],
            num_demos=[5],
        ),
    ],
    "Translation": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.mt.flores_101.{subset}",
            template=[
                    # "templates.translation.directed.simple",
                    # "templates.translation.directed.formal",
                    # "templates.translation.directed.casual"
                    "templates.translation.directed.blue_bench",
                # "templates.translation.directed.playful",
                # "templates.translation.directed.instructional",
                # "templates.translation.directed.title",
            ],
            num_demos=[5],
            demos_taken_from="validation",
        )
        for subset in subsets["flores_101"]
    ],

    "Chatbot_abilities": [
        _DefaultUnitxtRecipeArgs(
            card="cards.arena_hard.generation.english_gpt_4_0314_reference,metrics=[metrics.llm_as_judge.pairwise_comparative_rating.llama_3_70b_instruct_ibm_genai_template_arena_hard]",
            template=[
                "templates.empty",
            ],
            demos_pool_size=0,
            num_demos=[0],
        )
    ],
    "News_classification": [
        _DefaultUnitxtRecipeArgs(
            card="cards.20_newsgroups",
            template=[
                # "templates.classification.multi_class.instruction",
                # "templates.classification.multi_class.instruct_question_selects",
                # "templates.classification.multi_class.instruct_question_select_i_think",
                # "templates.classification.multi_class.instruct_select_question",
                "templates.classification.multi_class.blue_bench",
            ],
            num_demos=[1],
        ),
    ],
    "Bias": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.safety.bbq.{subset}",
            template=[
                "templates.qa.multiple_choice.with_context.match"
            ],  # TODO: only one template, contacted @jb about it
            demos_pool_size=20,
            demos_taken_from="test",  # This is not cool but there is no train
            num_demos=5,
        )
        for subset in subsets["bbq"]
    ],
    "Legal": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.legalbench.{subset}",
            template=[
                # "templates.classification.multi_class.instruction",
                # "templates.classification.multi_class.instruct_question_selects",
                # "templates.classification.multi_class.instruct_question_select_i_think",
                # "templates.classification.multi_class.instruct_select_question",
                "templates.classification.multi_class.blue_bench",
            ],
            demos_pool_size=10,
            demos_taken_from="test",  # TODO: this is not cool but the train is super small
            num_demos=1,
        )
        for subset in subsets["legalbench"]
    ],
    "Product_help": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.CFPB.product.{subset}",
            template=[
                # "templates.classification.multi_class.instruct_question_selects",
                # "templates.classification.multi_class.instruct_question_select_i_think",
                # "templates.classification.multi_class.instruct_select_question",
                "templates.classification.multi_class.blue_bench",
            ],
            num_demos=5,
        )
        for subset in subsets["CFPB"]
    ],
    "Knowledge": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.mmlu_pro.{subset}",
            template=[
                # "templates.qa.multiple_choice.with_topic.fm_eval",
                # "templates.qa.multiple_choice.with_topic.mmlu",
                # "templates.qa.multiple_choice.with_topic.helm",
                "templates.qa.multiple_choice.blue_bench",
                # "templates.qa.multiple_choice.with_topic.lm_eval_harness",
                # "templates.qa.multiple_choice.with_topic.title",
                # "templates.qa.multiple_choice.with_topic.match",
            ],
            demos_pool_size=20,
            demos_taken_from="test",  # TODO: this is not cool but there is no train
            num_demos=5,
        )
        for subset in subsets["mmlu_pro"]
    ],
    "Entity_extraction": [
        _DefaultUnitxtRecipeArgs(
            card=f"cards.universal_ner.{subset},metrics=[metrics.ner[zero_division=1.0]]",
            template=["templates.span_labeling.extraction.title"],
            train_refiner="operators.balancers.ner.zero_vs_many_entities[segments_boundaries=[0,1,2]]",
            demos_taken_from=(
                "test"  # TODO: this is not cool but there is no train
                if "pud" in subset
                else "train"
            ),
            num_demos=5,
            demos_pool_size=10000,
        )
        for subset in subsets["universal_ner"]
    ],
    "Safety": [
        _DefaultUnitxtRecipeArgs(
            card="cards.attaq_500",
            template_card_index=[0],  # no need for more
            demos_pool_size=0,
            num_demos=[0],
        ),
    ],
    "Summarization": [
        _DefaultUnitxtRecipeArgs(
            card="cards.billsum_document_filtered_to_6000_chars",
            template=[
                # "templates.summarization.abstractive.instruct_full",
                # "templates.summarization.abstractive.instruct_one_sentence",
                # "templates.summarization.abstractive.instruct_passive",
                "templates.summarization.abstractive.blue_bench",
                # "templates.summarization.abstractive.instruct_write_succinct",
                # "templates.summarization.abstractive.instruct_tldr",
            ],
            num_demos=[0],
        ),
        _DefaultUnitxtRecipeArgs(
            card="cards.tldr_document_filtered_to_6000_chars",
            template=[
                # "templates.summarization.abstractive.instruct_full",
                # "templates.summarization.abstractive.instruct_one_sentence",
                # "templates.summarization.abstractive.instruct_passive",
                "templates.summarization.abstractive.blue_bench",
                # "templates.summarization.abstractive.instruct_write_succinct",
                # "templates.summarization.abstractive.instruct_tldr",
            ],
            num_demos=[0],
        ),
    ],
    "RAG_general": [
        _DefaultUnitxtRecipeArgs(
            card="cards.rag.response_generation.clapnq",
            template=[
                # "templates.rag.response_generation.please_respond",
                # "templates.rag.response_generation.answer_based_on_context",
                # "templates.rag.response_generation.answer_based_on_context_inverted",
                "templates.rag.response_generation.blue_bench",
            ],
            num_demos=[1],
        )
    ],
    "RAG_finance": [
        _DefaultUnitxtRecipeArgs(
            card="cards.fin_qa",
            # template=[
                # "templates.finqa.default",
                # "templates.qa.with_context.simple2",
                # "templates.qa.with_context.with_type",
                # "templates.qa.with_context.question_first",
            # ],
            template_card_index=[0],
            num_demos=[2],  # TODO: add more
            demos_pool_size=10,
        ),
    ],
}

long_output_tasks = [
    "cards.billsum_document_filtered_to_6000_chars",
    "cards.tldr_document_filtered_to_6000_chars",
    "cards.arena_hard.generation.english_gpt_4_0314_reference,metrics=[metrics.llm_as_judge.pairwise_comparative_rating.llama_3_70b_instruct_ibm_genai_template_arena_hard]",
]

bench_cards = [
    uni_args.card
    for uni_args_list in unitxt_recipe_args_by_groupings.values()
    for uni_args in uni_args_list
]

assert all([long_output_task in bench_cards for long_output_task in long_output_tasks])

if __name__ == "__main__":
    BLUEBNECH_YAML = os.path.join(os.path.dirname(__file__), '_bluebench.yaml')
    print("BLUEBNECH_YAML:", BLUEBNECH_YAML)

    with open(BLUEBNECH_YAML, "w") as file:
        file.write("group: bluebench\ntask:\n")
        # print("#### Groups\n")
        print("#### Tasks\n")
        for group in unitxt_recipe_args_by_groupings:
            file.write(f'- group: "bluebench_{group}"\n  task:\n')
            # print(f'* `bluebench_{group}`')

            group_file_name = os.path.join(os.path.dirname(__file__), f'_bluebench_{group}.yaml')
            with open(group_file_name, "w") as group_file:
                group_file.write(f'group: bluebench_{group}\n')
                group_file.write(f'task:\n')

                for task in unitxt_recipe_args_by_groupings[group]:
                    task_name = task.card[len("cards."):]
                    if task.card.find(",") > 0:
                        task_name = task.card[:task.card.find(",")]
                    task_name = task_name.replace(".", "_")
                    file.write(f'    - "bluebench_{group}_{task_name}"\n')
                    group_file.write(f'  - "bluebench_{group}_{task_name}"\n')

                    current_file = os.path.join(os.path.dirname(__file__), f'bluebench_{group}_{task_name}.yaml')
                    print(f'* `bluebench_{group}_{task_name}`')
                    with open(current_file, "w") as task_file:
                        task_file.write(f'task: bluebench_{group}_{task_name}\n')
                        task_file.write(f'include: unitxt\n')
                        task_file.write(f'recipe: card={task.card},')
                        if len(task.template) > 1:
                            task_file.write(f'template=[{",".join(task.template)}],')
                            # task_file.write(f'template=[{task.template[0]}],')
                        elif len(task.template) > 0:
                            task_file.write(f'template={task.template[0] if type(task.template) == list and len(task.template) > 0 else ""},')
                        task_file.write(f'demos_pool_size={task.demos_pool_size[0] if type(task.demos_pool_size) == list else task.demos_pool_size},')
                        if type(task.train_refiner) == list and len(task.train_refiner) > 0:
                            task_file.write(f'train_refiner={task.train_refiner[0]},')
                        if len(task.template) == 0:
                            task_file.write(f'template_card_index={task.template_card_index[0] if type(task.template_card_index) == list else task.template_card_index},')
                        task_file.write(f'demos_taken_from={task.demos_taken_from[0] if type(task.demos_taken_from) == list else task.demos_taken_from}\n')
                        task_file.write(f'metadata:\n  version: 0')
                        # task_file.write(f'system_prompt={task.system_prompt[0] if type(task.system_prompt) == list else task.system_prompt},')
                        # task_file.write(f'format={task.format[0] if type(task.format) == list else task.format}')
                group_file.write(f'metadata:\n  version: 0')
        file.write(f'metadata:\n  version: 0')