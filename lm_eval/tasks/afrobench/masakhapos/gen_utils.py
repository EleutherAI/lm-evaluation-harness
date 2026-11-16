import argparse
import os

import yaml


class FunctionTag:
    def __init__(self, value):
        self.value = value


def prompt_func(mode, lang):
    prompt_map = {
        "prompt_1": "Please provide the POS tags for each word in the input sentence. The input will be a list of "
        "words in the sentence. The output format should be a list of tuples, where each tuple consists of "
        "a word from the input text and its corresponding POS tag label from the tag label set: ['ADJ', "
        "'ADP', 'ADV', 'AUX', 'CCONJ, 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', "
        "'SCONJ', 'SYM', 'VERB', 'X']. \nYour response should include only a list of tuples, in the order "
        "that the words appear in the input sentence, including punctuations, with each tuple containing the corresponding POS tag "
        "label for a word. \n\nSentence: {{tokens}} \nOutput: ",
        "prompt_2": f"You are an expert in tagging words and sentences in {lang} with the right POS tag. "
        f"\n\nPlease provide the POS tags for each word in the {lang} sentence. The input is a list of words in"
        " the sentence. POS tag label set: ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ, 'DET', 'INTJ', 'NOUN', "
        "'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']. The output format should "
        "be a list of tuples, where each tuple consists of a word from the input text and its corresponding"
        " POS tag label from the POS tag label set provided\nYour response should include only a list of "
        "tuples, in the order that the words appear in the input sentence, including punctuations, with each tuple containing the "
        "corresponding POS tag label for a word. \n\nSentence: {{tokens}} \nOutput: ",
        "prompt_3": f"Acting as a {lang} linguist and without making any corrections or changes to the text, perform a part of "
        "speech (POS) analysis of the sentences using the following POS tag label annotation ['ADJ', "
        "'ADP', 'ADV', 'AUX', 'CCONJ, 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', "
        "'SCONJ', 'SYM', 'VERB', 'X']. The input will be a list of words in the sentence. The output format should "
        "be a list of tuples, where each tuple consists of a word from the input text and its corresponding"
        " POS tag label from the POS tag label set provided\nYour response should include only a list of "
        "tuples, in the order that the words appear in the input sentence, including punctuations, with each tuple containing the "
        "corresponding POS tag label for a word. \n\nSentence: {{tokens}} \nOutput: ",
        "prompt_4": "Annotate each word in the provided sentence with the appropriate POS tag. The annotation "
        "list is given as: ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ, 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', "
        "'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']. The input sentence will be a list of words"
        " in the sentence. The output format should "
        "be a list of tuples, where each tuple consists of a word from the input text and its corresponding"
        " POS tag label from the POS tag label set provided\nYour response should include only a list of "
        "tuples, in the order that the words appear in the input sentence, including punctuations, with each tuple containing the "
        "corresponding POS tag label for a word. \n\nSentence: {{tokens}} \nOutput: ",
        "prompt_5": "Given the following sentence, identify the part of speech (POS) for each word. Use the following "
        "POS tag set: \nNOUN: Noun (person, place, thing), \nVERB: Verb (action, state), "
        "\nADJ: Adjective (describes a noun), \nADV: Adverb (modifies a verb, adjective, or adverb), "
        "\nPRON: Pronoun (replaces a noun), \nDET: Determiner (introduces a noun), "
        "\nADP: Adposition (preposition or postposition), \nCCONJ: Conjunction (connects words, phrases, clauses)"
        "\nPUNCT: Punctuation, \nPROPN: Proper Noun, \nAUX: Auxiliary verb (helper verb), "
        "\nSCONJ: Subordinating conjunction \nPART: Particle, \nSYM: Symbol, \nINTJ: Interjection, "
        "\nNUM: Numeral, \nX: others. The output format should "
        "be a list of tuples, where each tuple consists of a word from the input text and its corresponding"
        " POS tag label key only from the POS tag set provided\nYour response should include only a list of "
        "tuples, in the order that the words appear in the input sentence, including punctuations, with each tuple containing the "
        "corresponding POS tag label for a word. \n\nSentence: {{tokens}} \nOutput: ",
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
        "bam": "Bambara",
        "bbj": "Ghomala",
        "ewe": "Ewe",
        "fon": "Fon",
        "hau": "Hausa",
        "ibo": "Igbo",
        "kin": "Kinyarwanda",
        "lug": "Luganda",
        "luo": "Dholuo",
        "mos": "Mossi",
        "nya": "Chichewa",
        "pcm": "Nigerian Pidgin",
        "sna": "chiShona",
        "swa": "Kiswahili",
        "tsn": "Setswana",
        "twi": "Twi",
        "wol": "Wolof",
        "xho": "isiXhosa",
        "yor": "Yoruba",
        "zul": "isiZulu",
    }

    for lang in languages.keys():
        try:
            file_name = f"masakhapos_{lang}.yaml"
            task_name = f"masakhapos_{lang}_{mode}"
            yaml_template = "masakhapos_yaml"
            yaml_details = {
                "include": yaml_template,
                "task": task_name,
                "dataset_name": lang,
                "doc_to_text": prompt_func(mode, languages[lang]),
            }
            os.makedirs(f"{output_dir}/{mode}", exist_ok=True)
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
        default="prompt_1",
        choices=["prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5"],
        help="Prompt number",
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite, mode=args.mode)


if __name__ == "__main__":
    main()
