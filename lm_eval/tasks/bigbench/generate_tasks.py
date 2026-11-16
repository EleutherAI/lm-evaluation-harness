import os

import datasets
import yaml


all_subtasks = [
    "abstract_narrative_understanding",
    "anachronisms",
    "analogical_similarity",
    "analytic_entailment",
    "arithmetic",
    "ascii_word_recognition",
    "authorship_verification",
    "auto_categorization",
    "auto_debugging",
    "bbq_lite_json",
    "bridging_anaphora_resolution_barqa",
    "causal_judgment",
    "cause_and_effect",
    "checkmate_in_one",
    "chess_state_tracking",
    "chinese_remainder_theorem",
    "cifar10_classification",
    "code_line_description",
    "codenames",
    "color",
    "common_morpheme",
    "conceptual_combinations",
    "conlang_translation",
    "contextual_parametric_knowledge_conflicts",
    "crash_blossom",
    "crass_ai",
    "cryobiology_spanish",
    "cryptonite",
    "cs_algorithms",
    "dark_humor_detection",
    "date_understanding",
    "disambiguation_qa",
    "discourse_marker_prediction",
    "disfl_qa",
    "dyck_languages",
    "elementary_math_qa",
    "emoji_movie",
    "emojis_emotion_prediction",
    "empirical_judgments",
    "english_proverbs",
    "english_russian_proverbs",
    "entailed_polarity",
    "entailed_polarity_hindi",
    "epistemic_reasoning",
    "evaluating_information_essentiality",
    "fact_checker",
    "fantasy_reasoning",
    "few_shot_nlg",
    "figure_of_speech_detection",
    "formal_fallacies_syllogisms_negation",
    "gem",
    "gender_inclusive_sentences_german",
    "general_knowledge",
    "geometric_shapes",
    "goal_step_wikihow",
    "gre_reading_comprehension",
    "hhh_alignment",
    "hindi_question_answering",
    "hindu_knowledge",
    "hinglish_toxicity",
    "human_organs_senses",
    "hyperbaton",
    "identify_math_theorems",
    "identify_odd_metaphor",
    "implicatures",
    "implicit_relations",
    "intent_recognition",
    "international_phonetic_alphabet_nli",
    "international_phonetic_alphabet_transliterate",
    "intersect_geometry",
    "irony_identification",
    "kanji_ascii",
    "kannada",
    "key_value_maps",
    "known_unknowns",
    "language_games",
    "language_identification",
    "linguistic_mappings",
    "linguistics_puzzles",
    "list_functions",
    "logic_grid_puzzle",
    "logical_args",
    "logical_deduction",
    "logical_fallacy_detection",
    "logical_sequence",
    "mathematical_induction",
    "matrixshapes",
    "metaphor_boolean",
    "metaphor_understanding",
    "minute_mysteries_qa",
    "misconceptions",
    "misconceptions_russian",
    "mnist_ascii",
    "modified_arithmetic",
    "moral_permissibility",
    "movie_dialog_same_or_different",
    "movie_recommendation",
    "mult_data_wrangling",
    "multiemo",
    "natural_instructions",
    "navigate",
    "nonsense_words_grammar",
    "novel_concepts",
    "object_counting",
    "odd_one_out",
    "operators",
    "paragraph_segmentation",
    "parsinlu_qa",
    "parsinlu_reading_comprehension",
    "penguins_in_a_table",
    "periodic_elements",
    "persian_idioms",
    "phrase_relatedness",
    "physical_intuition",
    "physics",
    "physics_questions",
    "play_dialog_same_or_different",
    "polish_sequence_labeling",
    "presuppositions_as_nli",
    "qa_wikidata",
    "question_selection",
    "real_or_fake_text",
    "reasoning_about_colored_objects",
    "repeat_copy_logic",
    "rephrase",
    "riddle_sense",
    "ruin_names",
    "salient_translation_error_detection",
    "scientific_press_release",
    "semantic_parsing_in_context_sparc",
    "semantic_parsing_spider",
    "sentence_ambiguity",
    "similarities_abstraction",
    "simp_turing_concept",
    "simple_arithmetic_json",
    "simple_arithmetic_json_multiple_choice",
    "simple_arithmetic_json_subtasks",
    "simple_arithmetic_multiple_targets_json",
    "simple_ethical_questions",
    "simple_text_editing",
    "snarks",
    "social_iqa",
    "social_support",
    "sports_understanding",
    "strange_stories",
    "strategyqa",
    "sufficient_information",
    "suicide_risk",
    "swahili_english_proverbs",
    "swedish_to_german_proverbs",
    "symbol_interpretation",
    "temporal_sequences",
    "tense",
    "timedial",
    "topical_chat",
    "tracking_shuffled_objects",
    "understanding_fables",
    "undo_permutation",
    "unit_conversion",
    "unit_interpretation",
    "unnatural_in_context_learning",
    "vitaminc_fact_verification",
    "what_is_the_tao",
    "which_wiki_edit",
    "winowhy",
    "word_sorting",
    "word_unscrambling",
]

skip_tasks = [
    "simple_arithmetic_json_multiple_choice",
    "simple_arithmetic_multiple_targets_json",
]


def main() -> None:
    for path, task_type in zip(
        ["multiple_choice", "generate_until"],
        ["multiple_choice_template_yaml", "generate_until_template_yaml"],
    ):
        os.makedirs(path, exist_ok=True)
        for task in all_subtasks:
            file_name = f"{task}.yaml"
            try:
                template_file = task_type
                if path == "multiple_choice":
                    print(f"Checking {task} for multiple choices")
                    if task in skip_tasks:
                        continue
                    data = datasets.load_dataset("hails/bigbench", task + "_zero_shot")
                    multiple_choice_targets = data["default"][0][
                        "multiple_choice_targets"
                    ]
                    if len(multiple_choice_targets) == 0:
                        continue
                    else:
                        template_file = "multiple_choice_template_b_yaml"
                        if set(data["default"][0]["targets"]) < set(
                            multiple_choice_targets
                        ):
                            template_file = "multiple_choice_template_a_yaml"

                with open(f"{path}/{file_name}", "w", encoding="utf-8") as f:
                    f.write("# Generated by utils.py\n")
                    yaml.dump(
                        {
                            "include": f"../{template_file}",
                            "task": "bigbench_"
                            + task
                            + "_{}".format(task_type.split("_template_yaml")[0]),
                            "dataset_name": task
                            + "_zero_shot",  # zero-shot version of the dataset
                        },
                        f,
                        width=float("inf"),
                        allow_unicode=True,
                    )
            except FileExistsError:
                pass


if __name__ == "__main__":
    main()
