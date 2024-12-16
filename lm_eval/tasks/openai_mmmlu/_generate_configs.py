"""
Caveats:
  * I asked an LLM to translate "Answer:" into the 14 languages following, output of its conversation below.
  * OpenAI uses slightly different formatting for Q/A -> https://github.com/openai/simple-evals/blob/main/common.py
    a. They use a list of regexps in various langauages to extract a line that contains "Answer: .." to get the response,
    b. They have an actual prompt `QUERY_TEMPLATE_MULTICHOICE` with cot priming - I added this.

======
Can you make me a dictionary that maps these languages to the way one would say "Answer:" in those languages? Let me know of when you have doubts and why

LANGUAGES = {
  "AR_XY": "Arabic (Generic)",
  "BN_BD": "Bengali (Bangladesh)",
  "DE_DE": "German (Germany)",
  "ES_LA": "Spanish (Latin America)",
  "FR_FR": "French (France)",
  "HI_IN": "Hindi (India)",
  "ID_ID": "Indonesian (Indonesia)",
  "IT_IT": "Italian (Italy)",
  "JA_JP": "Japanese (Japan)",
  "KO_KR": "Korean (South Korea)",
  "PT_BR": "Portuguese (Brazil)",
  "ZH_CN": "Chinese (China)",
  "SW_KE": "Swahili (Kenya)",
  "YO_NG": "Yoruba (Nigeria)"
}

I'll create a dictionary that maps these languages to their respective translations of "Answer:". I'll think through this step-by-step and highlight any uncertainties.
    First, I'll create a new dictionary with the same keys as the LANGUAGES dictionary.
    For each language, I'll provide the translation of "Answer:".
    I'll note any doubts or uncertainties as I go.

Here's the process:
    AR_XY (Arabic): "الإجابة:" No doubts here.
    BN_BD (Bengali): "উত্তর:" No doubts.
    DE_DE (German): "Antwort:" Confident about this one.
    ES_LA (Spanish): "Respuesta:" Note: This is standard Spanish, but there might be slight variations in Latin American countries.
    FR_FR (French): "Réponse :" Note: In French, there's usually a space before the colon.
    HI_IN (Hindi): "उत्तर:" No doubts.
    ID_ID (Indonesian): "Jawaban:" Fairly confident, but there might be variations.
    IT_IT (Italian): "Risposta:" No doubts.
    JA_JP (Japanese): "答え：" Note: Japanese uses a full-width colon (：) instead of a half-width one (:).
    KO_KR (Korean): "답변:" Fairly confident, but there might be other ways to express this.
    PT_BR (Portuguese): "Resposta:" No doubts for Brazilian Portuguese.
    ZH_CN (Chinese): "答案：" Note: Using Simplified Chinese characters. Also uses full-width colon.
    SW_KE (Swahili): "Jibu:" Some uncertainty here, as my knowledge of Swahili is not as extensive.
    YO_NG (Yoruba): "Ìdáhùn:" Significant uncertainty. My knowledge of Yoruba is limited, and this might not be the most common or accurate translation.

Main points of uncertainty:
    For ES_LA (Spanish), there might be slight regional variations in Latin America.
    For SW_KE (Swahili) and YO_NG (Yoruba), my confidence is lower due to limited knowledge of these languages.
    For some languages, there might be multiple ways to express "Answer:" depending on context or formality level.

It would be advisable to have these translations verified by native speakers or language experts, especially for Swahili and Yoruba.
======


"""

import argparse
import logging
from pathlib import Path

import yaml
from tqdm import tqdm
from utils import ANSWERS, LANGUAGES, SUBJECTS


eval_logger = logging.getLogger("lm-eval")


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    PROMPT_FLAVOURS = {
        # default in the version of standard MMLU
        # honestly I think we should translate these into the target language.
        "default": {
            "description": "The following are multiple choice questions (with answers) about {}.\n\n",
            "prompt": "{{Question.strip()}}\nA. {{A.strip()}}\nB. {{B.strip()}}\nC. {{C.strip()}}\nD. {{D.strip()}}\n",
            "add_answer": True,
        },
        # this one in the version found on simple-evals from openai
        # "cot": {
        #     "description": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n",
        #     "prompt": "{{Question.strip()}}\n\nA) {{A.strip()}}\nB) {{B.strip()}}\nC) {{C.strip()}}\nD) {{D.strip()}}\n",
        #     "add_answer": False
        # }
    }

    ALL_CATEGORIES = []
    ALL_TASKS = []
    for prompt_key, prompt_info in PROMPT_FLAVOURS.items():
        for langgode, language_full_name in tqdm(LANGUAGES.items()):
            _langgode = langgode.lower()
            out_folder = Path(prompt_key) / _langgode
            out_folder.mkdir(exist_ok=True, parents=True)
            for subject, category in SUBJECTS.items():
                if category not in ALL_CATEGORIES:
                    ALL_CATEGORIES.append(category)

                yaml_dict = {
                    "include": "../../_default_template_yaml",
                    "tag": f"openai_mmmlu_{prompt_key}_{_langgode}_{category}",
                    "task": f"openai_mmmlu_{prompt_key}_{_langgode}_{subject}",
                    "task_alias": f'{_langgode} {subject.replace("_", " ")}',
                    "dataset_name": subject,
                    "test_split": langgode,
                    "description": prompt_info["description"].format(subject),
                    "doc_to_text": prompt_info["prompt"]
                    + (ANSWERS[langgode] if prompt_info["add_answer"] else ""),
                    "doc_to_choice": ["A", "B", "C", "D"],
                    "doc_to_target": "{{Answer.strip()}}",
                }

                file_save_path = (
                    out_folder / f"openai_mmmlu_{prompt_key}_{subject}.yaml"
                )
                eval_logger.info(
                    f"Saving yaml for subset {_langgode},{subject} to {file_save_path}"
                )
                with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                    yaml.dump(
                        yaml_dict,
                        yaml_file,
                        allow_unicode=True,
                        default_style='"',
                    )

            # (sub)group for prompt/language pair
            subgroup_info_path = (
                out_folder / f"_{prompt_key}_{_langgode}_group_info.yaml"
            )
            with open(subgroup_info_path, "w", encoding="utf-8") as yaml_file:
                # list of task for this pair of prompt/language
                _tasks = [
                    f"openai_mmmlu_{prompt_key}_{_langgode}_{_subject}"
                    for _subject in SUBJECTS.keys()
                ]
                dct = {
                    "group": f"openai_mmmlu_{prompt_key}_{_langgode}",
                    "task": _tasks,
                    "aggregate_metric_list": [
                        {"metric": "acc", "weight_by_size": True}
                    ],
                    "metadata": {"version": "1.0.0"},
                }
                ALL_TASKS.extend(_tasks)
                yaml.dump(
                    dct,
                    yaml_file,
                    indent=4,
                    default_flow_style=False,
                )
        # (super)group for promptkey
        out_folder = Path(prompt_key)
        supergroup_info_path = out_folder / f"_{prompt_key}_group_info.yaml"
        with open(supergroup_info_path, "w", encoding="utf-8") as yaml_file:
            dct = {
                "group": f"openai_mmmlu_{prompt_key}",
                "task": ALL_TASKS,
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
                "metadata": {"version": "1.0.0"},
            }

            yaml.dump(
                dct,
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
