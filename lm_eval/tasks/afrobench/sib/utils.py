import argparse
import os

import yaml


class FunctionTag:
    def __init__(self, value):
        self.value = value


def prompt_func(mode, lang):
    prompt_map = {
        "prompt_1": "Given the categories science/technology, travel, politics, sports, health, entertainment, or geography; what category does the text: '{{text}}' belong to: \n\n",
        "prompt_2": f"Does this {lang} topic; "
        "'{{text}}' belong to one of the following categories: science/technology, travel, politics, sports, health, entertainment, or geography? category only\n\n",
        "prompt_3": f"You are an assistant able to classify topics in texts. \n\n"
        f"Given the categories science/technology, travel, politics, sports, health, entertainment, or geography; what is "
        f"the topic of the {lang} statement below? Return only the category. "
        "\n\ntext: {{text}} \\category:\n\n",
        "prompt_4": "Label the following text as science/technology, travel, politics, sports, health, entertainment, or geography. Provide only the category as your "
        "response. \n\ntext: {{text}} \\category: \n\n",
        "prompt_5": f"You are tasked with performing topic classification on the following {lang} text. "
        f"For each input, classify the topic as science/technology, travel, politics, sports, health, entertainment, or geography. "
        f"Use the following guidelines: \n\n "
        f"science/technology: The text discusses scientific discoveries, technological advancements, or related topics. \n"
        f"travel: The text describes travel experiences, destinations, or related topics. \n"
        f"politics: The text covers political events, policies, or related topics. \n"
        f"sports: The text talks about sports events, athletes, or related topics. \n"
        f"health: The text addresses health issues, medical advancements, or related topics. \n"
        f"entertainment: The text pertains to movies, music, celebrities, or related topics. \n"
        f"geography: The text involves geographical information, locations, or related topics. \n\n"
        f"If the text contains multiple topics, choose the dominant topic. "
        f"For ambiguous or unclear topics, select the category that best reflects the overall content. "
        "Please provide a single classification for each input.\n\ntext: {{text}} \\category: \n\n",
    }
    return prompt_map[mode]


def gen_lang_yamls(output_dir: str, overwrite: bool, mode: str) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    languages = {
        "aeb": "Tunisian Arabic",
        "afr": "Afrikaans",
        "aka": "Akan",
        "amh": "Amharic",
        "ary": "Moroccan Arabic",
        "arz": "Egyptian Arabic",
        "bam": "Bambara",
        "bem": "Bemba",
        "cjk": "Chokwe",
        "dik": "Southwestern Dinka",
        "dyu": "Dyula",
        "eng": "English",
        "ewe": "Ewe",
        "fon": "Fon",
        "fra": "French",
        "fuv": "Nigerian Fulfulde",
        "gaz": "West Central Oromo",
        "hau": "Hausa",
        "ibo": "Igbo",
        "kab": "Kabyle",
        "kam": "Kamba",
        "kmb": "Kimbundu",
        "kbp": "Kabiye",
        "kea": "Kabuverdianu",
        "kik": "Kikuyu",
        "kin": "Kinyarwanda",
        "kon": "Kikongo",
        "knc": "Central Kanuri",
        "lua": "Luba-Kasai",
        "lug": "Luganda",
        "luo": "Luo",
        "lin": "Lingala",
        "mos": "Mossi",
        "nus": "Nuer",
        "nso": "Northern Sotho",
        "nya": "Nyanga",
        "plt": "Plateau Malagasy",
        "por": "Portuguese",
        "run": "Rundi",
        "sag": "Sango",
        "sna": "Shona",
        "som": "Somali",
        "sot": "Southern Sotho",
        "ssw": "Swazi",
        "swa": "Swahili",
        "taq": "Tamasheq",
        "tir": "Tigrinya",
        "tum": "Tumbuka",
        "tso": "Tsonga",
        "twi": "Twi",
        "tzm": "Tamazight",
        "umb": "Umbundu",
        "wol": "Wolof",
        "xho": "Xhosa",
        "yor": "Yoruba",
        "zul": "Zulu",
    }

    lang_2_dataset_lang_code = {
        "aeb": "aeb_Arab",
        "afr": "afr_Latn",
        "aka": "aka_Latn",
        "amh": "amh_Ethi",
        "ary": "ary_Arab",
        "arz": "arz_Arab",
        "bam": "bam_Latn",
        "bem": "bem_Latn",
        "cjk": "cjk_Latn",
        "dik": "dik_Latn",
        "dyu": "dyu_Latn",
        "eng": "eng_Latn",
        "ewe": "ewe_Latn",
        "fon": "fon_Latn",
        "fra": "fra_Latn",
        "fuv": "fuv_Latn",
        "gaz": "gaz_Latn",
        "hau": "hau_Latn",
        "ibo": "ibo_Latn",
        "kab": "kab_Latn",
        "kam": "kam_Latn",
        "kmb": "kmb_Latn",
        "kbp": "kbp_Latn",
        "kea": "kea_Latn",
        "kik": "kik_Latn",
        "kin": "kin_Latn",
        "kon": "kon_Latn",
        "knc": "knc_Latn",
        "lua": "lua_Latn",
        "lug": "lug_Latn",
        "luo": "luo_Latn",
        "lin": "lin_Latn",
        "mos": "mos_Latn",
        "nus": "nus_Latn",
        "nso": "nso_Latn",
        "nya": "nya_Latn",
        "plt": "plt_Latn",
        "por": "por_Latn",
        "run": "run_Latn",
        "sag": "sag_Latn",
        "sna": "sna_Latn",
        "som": "som_Latn",
        "sot": "sot_Latn",
        "ssw": "ssw_Latn",
        "swa": "swh_Latn",
        "taq": "taq_Latn",
        "tir": "tir_Ethi",
        "tum": "tum_Latn",
        "tso": "tso_Latn",
        "twi": "twi_Latn",
        "tzm": "tzm_Tfng",
        "umb": "umb_Latn",
        "wol": "wol_Latn",
        "xho": "xho_Latn",
        "yor": "yor_Latn",
        "zul": "zul_Latn",
    }

    for lang in languages.keys():
        try:
            file_name = f"sib_{lang}.yaml"
            task_name = f"sib_{lang}_{mode}"
            yaml_template = "sib"
            yaml_details = {
                "include": yaml_template,
                "task": task_name,
                "dataset_name": lang_2_dataset_lang_code[lang],
                "doc_to_text": prompt_func(mode, languages[lang]),
            }
            file_path = os.path.join(output_dir, mode)
            os.makedirs(file_path, exist_ok=True)

            with open(
                f"{output_dir}/{mode}/{file_name}",
                "w" if overwrite else "x",
                encoding="utf8",
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    yaml_details,
                    f,
                    allow_unicode=True,
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Directory to write yaml files to",
    )
    parser.add_argument(
        "--mode",
        default="prompt_3",
        choices=["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"],
        help="Prompt number",
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite, mode=args.mode)


if __name__ == "__main__":
    main()
