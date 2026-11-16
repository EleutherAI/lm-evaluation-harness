"""
Copyright 2026 R3 Lab, University of Toronto.

Utility functions for processing Toksuite results and generating LaTeX tables for robustness analysis.


"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="toksuite-%(asctime)s - %(levelname)s - %(message)s"
)

LATEX_TABLE_CATEGORIES = {
    "Input (Non-EN)": [
        "toksuite_turkish_english_keyboard",
        "toksuite_italian_english_keyboard",
        "toksuite_farsi_arabic_keyboard_for_farsi",
        "toksuite_farsi_number_romanization",
        "toksuite_farsi_romanization",
        # "toksuite_chinese_partially_romanized"
        "toksuite_chinese_romanization",
        "toksuite_chinese_traditional",
    ],
    "Diacritics (Non-EN)": [
        "toksuite_chinese_optional_diacritics",
        "toksuite_farsi_optional_diacritics",
    ],
    "Orthographic & Grammatical Errors (EN)": [
        "toksuite_english_orthographic_errors",
        "toksuite_english_grammatical_errors",
    ],
    "Orthographic & Grammatical Errors (Non-EN)": [
        "toksuite_turkish_orthographic_errors",
        "toksuite_turkish_grammatical_errors",
        "toksuite_italian_orthographic_errors",
        "toksuite_italian_grammatical_errors",
    ],
    "Morphological (EN)": [
        "toksuite_english_contractions",
        "toksuite_english_inflections",
    ],
    "Morphological (Non-EN)": [
        "toksuite_italian_contractions",
        "toksuite_turkish_derivations",
        "toksuite_turkish_inflections",
    ],
    "Noise (EN)": [
        "toksuite_english_keyboard_proximity_errors",
        "toksuite_english_ocr_errors",
        "toksuite_stem_character_deletion",
        "toksuite_english_character_deletion",
        "toksuite_english_space_removal",
        "toksuite_math_space_removal",
        "toksuite_stem_space_removal",
        "toksuite_stem_typographical_errors",
    ],
    "Noise (Non-EN)": [
        "toksuite_italian_plausible_diacritics_errors",
        "toksuite_italian_keyboard_proximity_errors",
        "toksuite_farsi_keyboard_proximity_errors",
        "toksuite_turkish_keyboard_proximity_errors",
        "toksuite_chinese_keyboard_proximity_errors",
        "toksuite_chinese_ocr_errors",
        "toksuite_chinese_space_removal",
        "toksuite_turkish_typographical_errors",
        "toksuite_italian_typographical_errors",
        "toksuite_chinese_word_spacing_zero-width_characters_extra_space",
        "toksuite_farsi_word_spacing_zero-width_characters_extra_space",
    ],
    "LaTeX": [
        "toksuite_stem_latex",
        "toksuite_math_latex",
    ],
    "STEM (EN)": [
        "toksuite_stem_unusual_formatting",
    ],
    "Unicode": [
        "toksuite_stem_fullwidth_characters",
        "toksuite_math_decorative_unicode",
        "toksuite_english_scripted_text",
        "toksuite_stem_double_struck",
        "toksuite_stem_enclosed_characters",
        "toksuite_stem_unicode_formatting",
    ],
}

LATEX_COLUMN_DATA = [
    [
        "Input",
        "Diacr.",
        r"Orth. \& Gram.",
        "<same>",
        "Morph",
        "<same>",
        "Noise",
        "<same>",
        "LaTeX",
        "STEM",
        "Unic",
        "Avg",
    ],
    ["NEN", "NEN", "EN", "NEN", "EN", "NEN", "EN", "NEN", "EN", "EN", "EN", ""],
]


LATEX_TABLE_CATEGORIES_EN_ONLY = {
    "Orthographic & Grammatical Errors (EN)": [
        "toksuite_english_orthographic_errors",
        "toksuite_english_grammatical_errors",
    ],
    "Morphological (EN)": [
        "toksuite_english_contractions",
        "toksuite_english_inflections",
    ],
    "Noise (EN)": [
        "toksuite_english_keyboard_proximity_errors",
        "toksuite_english_ocr_errors",
        "toksuite_stem_character_deletion",
        "toksuite_english_character_deletion",
        "toksuite_english_space_removal",
        "toksuite_math_space_removal",
        "toksuite_stem_space_removal",
        "toksuite_stem_typographical_errors",
    ],
    "LaTeX": [
        "toksuite_stem_latex",
        "toksuite_math_latex",
    ],
    "STEM (EN)": [
        "toksuite_stem_unusual_formatting",
    ],
    "Unicode": [
        "toksuite_stem_fullwidth_characters",
        "toksuite_math_decorative_unicode",
        "toksuite_english_scripted_text",
        "toksuite_stem_double_struck",
        "toksuite_stem_enclosed_characters",
        "toksuite_stem_unicode_formatting",
    ],
}

LATEX_TABLE_CATEGORIES_CANONICAL = {
    "EN": ["toksuite_english_canonical"],
    "FA": ["toksuite_farsi_canonical"],
    "IT": ["toksuite_italian_canonical"],
    "TR": ["toksuite_turkish_canonical"],
    "ZH": ["toksuite_chinese_canonical"],
    "MATH": ["toksuite_math_canonical"],
    "STEM": ["toksuite_stem_canonical"],
}
LATEX_COLUMN_DATA_EN_ONLY = [
    [
        r"Orth. \& Gram.",
        "Morph",
        "Noise",
        "LaTeX",
        "STEM",
        "Unic",
        "Avg",
    ],
]

TITLE = "Robustness under multilingual text perturbations. Values represent relative performance drop ($\\frac{\\text{Acc}_{\\text{can}} - \\text{Acc}_{\\text{pert}}}{\\text{Acc}_{\\text{can}}}$); lower values indicate greater robustness. Perturbation types: Input: non-native keyboard/romanization; Diacr.: optional diacritics; Orth. Errors: orthographic errors; Morph.: derivations/inflections/contractions; Noise: homoglyphs/OCR/typos/spacing; LaTeX: LaTeX-style math formatting; STEM: scientific diagrams and notations; Unic.: Unicode styling characters. NEN: non-English. \\textbf{\\textcolor{green!70!black}{Green}} and \\textcolor{red}{red} entries indicate notable robustness and fragility, respectively."


def get_grouped_results(
    results_dct: dict[str, Any],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
) -> dict[str, float]:
    """Given a results dict loaded from lm-eval results json, process the results and return a dict of grouped results for each category in `latex_table_categories`."""
    ## process results
    results = results_dct["results"]
    results = pd.DataFrame(results).transpose().reset_index(names="task")

    ## extract task configs
    configs = (
        pd.DataFrame(results_dct["configs"])
        .transpose()[["task", "metadata"]]
        .reset_index(drop=True)
    )
    configs["num_samples"] = configs["metadata"].apply(
        lambda x: x.get("num_samples", np.nan)
    )
    if configs["num_samples"].isnull().any():
        logging.info(
            "Warning: Some tasks are missing num_samples in metadata, which may lead to incorrect robustness calculations."
        )
    configs["canonical_task"] = configs["metadata"].apply(
        lambda x: x.get("canonical_task")
    )
    configs["task_pretty_name"] = configs["metadata"].apply(
        lambda x: x.get("task_pretty_name")
    )
    configs = configs.drop(columns=["metadata"])
    results = results.merge(configs, how="left", on="task")

    ## extract and compute robustness data
    latex_table_data = {}
    for category, tasks in latex_table_categories.items():
        category_results = results[results["task"].isin(tasks)]
        latex_table_data[category] = (
            category_results["acc_norm,none"] * category_results["num_samples"]
        ).sum() / category_results["num_samples"].sum()
    return latex_table_data


def get_grouped_results_df(
    results_files_to_process: list[Path | str],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
) -> pd.DataFrame:
    """Compute the robustness values for model tokenizer pairs given a list of lm-eval results files to process."""
    robustness_data = []
    for file in results_files_to_process:
        with open(file) as f:
            results_dct = json.load(f)
        ## add model information
        model_args = results_dct["config"]["model_args"]
        if isinstance(model_args, str):
            # e.g. "pretrained=toksuite/facebook-xglm-564M,tokenizer=facebook/xglm-564M",
            model_args = dict(item.split("=", 1) for item in model_args.split(","))
        model_name = model_args.get("pretrained")
        tokenizer_name = model_args.get("tokenizer", model_name)
        dct = get_grouped_results(results_dct, latex_table_categories)
        robustness_data.append(
            {
                "model_name": model_name,
                "tokenizer_name": tokenizer_name,
                **dct,
            }
        )
    robustness_df = pd.DataFrame(robustness_data)
    robustness_df["Avg"] = robustness_df.drop(
        columns=["model_name", "tokenizer_name"]
    ).mean(axis=1)
    return robustness_df


def get_robustness_dict_from_results(
    results_dct: dict[str, Any],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
) -> dict[str, float]:
    """Given a results dict loaded from lm-eval results json, process the results and return a dict of robustness values for each category in `latex_table_categories`."""
    ## process results
    results = results_dct["results"]
    # results has following columns: task	acc_norm,none	acc_norm_stderr,none	acc,none	acc_stderr,none	alias
    results = pd.DataFrame(results).transpose().reset_index(names="task")

    ## extract task configs
    configs = (
        pd.DataFrame(results_dct["configs"])
        .transpose()[["task", "metadata"]]
        .reset_index(drop=True)
    )
    configs["num_samples"] = configs["metadata"].apply(
        lambda x: x.get("num_samples", np.nan)
    )
    if configs["num_samples"].isnull().any():
        logging.info(
            "Warning: Some tasks are missing num_samples in metadata, which may lead to incorrect robustness calculations."
        )
    configs["canonical_task"] = configs["metadata"].apply(
        lambda x: x.get("canonical_task")
    )
    configs["task_pretty_name"] = configs["metadata"].apply(
        lambda x: x.get("task_pretty_name")
    )
    # configs has: task	metadata	num_samples	canonical_task	task_pretty_name
    configs = configs.drop(columns=["metadata"])
    results = results.merge(configs, how="left", on="task")
    results["canonical_task_acc_norm"] = results.apply(
        lambda row: results[results["task"] == row["canonical_task"]][
            "acc_norm,none"
        ].values[0]
        if row["canonical_task"] not in [None, np.nan]
        and not pd.isna(row["canonical_task"])
        else np.nan,
        axis=1,
    )

    ## extract and compute robustness data
    latex_table_data = {}
    for category, tasks in latex_table_categories.items():
        logging.debug(f"Processing category: {category} with tasks: {tasks}")
        category_results = results[results["task"].isin(tasks)]

        # get perturbed accuracy for each canonical_task
        def compute_weighted_avg(group: pd.DataFrame) -> float:
            return (group["acc_norm,none"] * group["num_samples"]).sum() / group[
                "num_samples"
            ].sum()

        avg_perturbed_acc = pd.DataFrame(
            category_results.groupby("canonical_task").apply(
                lambda x: (x["acc_norm,none"] * x["num_samples"]).sum()
                / x["num_samples"].sum(),
                include_groups=False,
            ),
            columns=["perturbed_acc_norm"],
        )
        canonical_acc = pd.DataFrame(
            category_results.groupby("canonical_task").apply(
                lambda x: x["canonical_task_acc_norm"].values[0],
                include_groups=False,
            ),
            columns=["canonical_acc_norm"],
        )
        intermediate_res = avg_perturbed_acc.merge(
            canonical_acc, left_index=True, right_index=True
        ).reset_index()
        intermediate_res["robustness"] = (
            intermediate_res["canonical_acc_norm"]
            - intermediate_res["perturbed_acc_norm"]
        ) / intermediate_res["canonical_acc_norm"]
        avg_robustness = intermediate_res["robustness"].mean()
        latex_table_data[category] = avg_robustness
    return latex_table_data


def get_robustness_df(
    results_files_to_process: list[Path | str],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
) -> pd.DataFrame:
    """Compute the robustness values for model tokenizer pairs given a list of lm-eval results files to process."""
    robustness_data = []
    for file in results_files_to_process:
        logging.info(f"Processing results file: {file}")
        with open(file) as f:
            results_dct = json.load(f)
        ## add model information
        model_args = results_dct["config"]["model_args"]
        if isinstance(model_args, str):
            # e.g. "pretrained=toksuite/facebook-xglm-564M,tokenizer=facebook/xglm-564M",
            model_args = dict(item.split("=", 1) for item in model_args.split(","))
        model_name = model_args.get("pretrained")
        tokenizer_name = model_args.get("tokenizer", model_name)
        robustness_dict = get_robustness_dict_from_results(
            results_dct, latex_table_categories
        )
        robustness_data.append(
            {
                "model_name": model_name,
                "tokenizer_name": tokenizer_name,
                **robustness_dict,
            }
        )
    robustness_df = pd.DataFrame(robustness_data)
    robustness_df["Avg"] = robustness_df.drop(
        columns=["model_name", "tokenizer_name"]
    ).mean(axis=1)
    robustness_df = robustness_df.sort_values(by="Avg", ascending=True)
    return robustness_df


def _get_canonical_performance_dict_from_results(
    results_dct: dict[str, Any],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
) -> dict[str, float]:
    """For each category return the mean accuracy of its unique canonical tasks."""
    results = pd.DataFrame(results_dct["results"]).transpose().reset_index(names="task")
    configs = (
        pd.DataFrame(results_dct["configs"])
        .transpose()[["task", "metadata"]]
        .reset_index(drop=True)
    )
    configs["canonical_task"] = configs["metadata"].apply(
        lambda x: x.get("canonical_task")
    )
    configs = configs.drop(columns=["metadata"])
    results = results.merge(configs, how="left", on="task")

    acc_lookup = results.set_index("task")["acc_norm,none"].to_dict()

    latex_table_data: dict[str, float] = {}
    for category, tasks in latex_table_categories.items():
        category_results = results[results["task"].isin(tasks)]
        canonical_tasks = category_results["canonical_task"].dropna().unique()
        canonical_accs = [acc_lookup[ct] for ct in canonical_tasks if ct in acc_lookup]
        latex_table_data[category] = (
            float(np.mean(canonical_accs)) if canonical_accs else np.nan
        )
    return latex_table_data


def _get_canonical_performance_df(
    results_files_to_process: list[Path | str],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
) -> pd.DataFrame:
    rows = []
    for file in results_files_to_process:
        logging.info(f"Processing results file: {file}")
        with open(file) as f:
            results_dct = json.load(f)
        model_args = results_dct["config"]["model_args"]
        if isinstance(model_args, str):
            model_args = dict(item.split("=", 1) for item in model_args.split(","))
        model_name = model_args.get("pretrained")
        tokenizer_name = model_args.get("tokenizer", model_name)
        dct = _get_canonical_performance_dict_from_results(
            results_dct, latex_table_categories
        )
        rows.append({"model_name": model_name, "tokenizer_name": tokenizer_name, **dct})
    df = pd.DataFrame(rows)
    df["Avg"] = df.drop(columns=["model_name", "tokenizer_name"]).mean(axis=1)
    df = df.sort_values(by="Avg", ascending=False)
    return df


OutputFormat = Literal["latex", "markdown", "dataframe"]

##########################################################
#################### Table Generation ####################


def _build_latex_table_str(
    df: pd.DataFrame,
    latex_column_data: list[list[str]],
    caption: str,
    label: str,
    higher_is_better: bool = True,
    color_extremes: bool = True,
) -> str:
    """Build a LaTeX tabularx table string from a DataFrame indexed by model name."""
    n_cols = len(df.columns)
    latex_str = f"""\\begin{{table}}[!htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\tiny
\\begin{{tabularx}}{{\\textwidth}}{{l|*{{{n_cols}}}{{>{{\\centering\\arraybackslash}}X}}}}
\\toprule
\\textbf{{Model}} &"""
    ## create multi-level headers if latex_column_data is provided
    for ind, level in enumerate(latex_column_data):
        if ind > 0:
            latex_str += "\\\\\n"
            latex_str += "&"
        i = 0
        while i < len(level):
            col = level[i]
            same_count = 1
            while i + same_count < len(level) and level[i + same_count] == "<same>":
                same_count += 1
            if ind == 0:
                latex_str += (
                    f" \\multicolumn{{{same_count}}}{{c}}{{\\textbf{{{col}}}}} &"
                )
            else:
                latex_str += f" \\multicolumn{{{same_count}}}{{c}}{{{col}}} &"
            i += same_count
        latex_str = latex_str.rstrip(" &")
    latex_str = latex_str.rstrip(" &") + " \\\\\n\\midrule\n"

    # Find best and worst values (according to higher_is_better) for each column to apply color formatting
    best_values, worst_values = {}, {}
    for col in df.columns:
        valid_vals = df[col].dropna()
        if len(valid_vals) > 1:
            try:
                best_values[col] = (
                    valid_vals.max() if higher_is_better else valid_vals.min()
                )
                worst_values[col] = (
                    valid_vals.min() if higher_is_better else valid_vals.max()
                )
            except Exception:
                pass

    # construct the body of the table
    for model in df.index:
        latex_str += model
        for col in df.columns:
            val = df.loc[model, col]
            if pd.isna(val):
                latex_str += " & ---"
            else:
                try:
                    val_str = f"{val:.2f}"
                    if color_extremes:
                        if col in best_values and abs(val - best_values[col]) < 1e-4:
                            val_str = f"\\textbf{{\\textcolor{{green!70!black}}{{{val_str}}}}}"
                        elif (
                            col in worst_values and abs(val - worst_values[col]) < 1e-4
                        ):
                            val_str = f"\\textcolor{{red}}{{{val_str}}}"
                except Exception:
                    val_str = str(val)
                latex_str += f" & {val_str}"
        latex_str = latex_str.rstrip("&")
        latex_str += " \\\\\n"

    # Create the summary row with column averages
    try:
        col_avgs = df.mean(axis=0, skipna=True)
        best_col_avg = col_avgs.max() if higher_is_better else col_avgs.min()
        worst_col_avg = col_avgs.min() if higher_is_better else col_avgs.max()
        latex_str += "\\midrule\nAvg"
        for col in df.columns:
            avg_val = col_avgs[col]
            if pd.isna(avg_val):
                latex_str += " & ---"
            else:
                try:
                    val_str = f"{avg_val:.2f}"
                    if color_extremes:
                        if abs(avg_val - best_col_avg) < 1e-6:
                            val_str = f"\\textbf{{\\textcolor{{green!70!black}}{{{val_str}}}}}"
                        elif abs(avg_val - worst_col_avg) < 1e-6:
                            val_str = f"\\textcolor{{red}}{{{val_str}}}"
                except Exception:
                    val_str = str(avg_val)
                latex_str += f" & {val_str}"
        latex_str += " \\\\\n"
    except Exception:
        pass

    latex_str += "\\bottomrule\n\\end{tabularx}\n\\end{table}"
    latex_str = latex_str.replace(r"\_", "ESCAPED_UNDERSCORE")
    latex_str = latex_str.replace("_", "\\_")
    latex_str = latex_str.replace("ESCAPED_UNDERSCORE", "\\_")
    return latex_str


def _build_markdown_table_str(
    df: pd.DataFrame,
    higher_is_better: bool = True,
    color_extremes: bool = True,
) -> str:
    """Build a Markdown table string from a DataFrame indexed by model name.

    Best values are **bolded**, worst values are *italicized* (when color_extremes=True).
    """
    cols = list(df.columns)

    best_values, worst_values = {}, {}
    if color_extremes:
        for col in cols:
            valid_vals = df[col].dropna()
            if len(valid_vals) > 1:
                try:
                    best_values[col] = (
                        valid_vals.max() if higher_is_better else valid_vals.min()
                    )
                    worst_values[col] = (
                        valid_vals.min() if higher_is_better else valid_vals.max()
                    )
                except Exception:
                    pass

    header = "| Model | " + " | ".join(cols) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, separator]

    for model in df.index:
        row = f"| {model} |"
        for col in cols:
            val = df.loc[model, col]
            if pd.isna(val):
                row += " --- |"
            else:
                try:
                    val_str = f"{val:.2f}"
                    if color_extremes:
                        if col in best_values and abs(val - best_values[col]) < 1e-4:
                            val_str = f"**{val_str}**"
                        elif (
                            col in worst_values and abs(val - worst_values[col]) < 1e-4
                        ):
                            val_str = f"*{val_str}*"
                except Exception:
                    val_str = str(val)
                row += f" {val_str} |"
        lines.append(row)

    try:
        col_avgs = df.mean(axis=0, skipna=True)
        best_col_avg = col_avgs.max() if higher_is_better else col_avgs.min()
        worst_col_avg = col_avgs.min() if higher_is_better else col_avgs.max()
        lines.append("| --- | " + " | ".join(["---"] * len(cols)) + " |")
        avg_row = "| Avg |"
        for col in cols:
            avg_val = col_avgs[col]
            if pd.isna(avg_val):
                avg_row += " --- |"
            else:
                try:
                    val_str = f"{avg_val:.2f}"
                    if color_extremes:
                        if abs(avg_val - best_col_avg) < 1e-6:
                            val_str = f"**{val_str}**"
                        elif abs(avg_val - worst_col_avg) < 1e-6:
                            val_str = f"*{val_str}*"
                except Exception:
                    val_str = str(avg_val)
                avg_row += f" {val_str} |"
        lines.append(avg_row)
    except Exception:
        pass

    return "\n".join(lines)


##########################################################
#################### Table String Generation ####################


def get_table_str(
    results_files_to_process: list[Path | str],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
    latex_column_data: list[list[str]] = LATEX_COLUMN_DATA,
    color_extremes: bool = True,
    output_format: OutputFormat = "latex",
) -> str | pd.DataFrame:
    """Generate formatted table for robustness drops.

    column_data could be used for n-level headers e.g. [["Input", "<same>"], ["Non-EN", "EN"]] -> this will produce (Input combining two columns) // Non-EN & EN
    """
    logging.info(
        f"Processing {len(results_files_to_process)} results files to compute robustness dataframe..."
    )

    robustness_df = get_robustness_df(results_files_to_process, latex_table_categories)
    robustness_df = robustness_df.set_index("model_name")
    ## If you want to report the tokenizer, don't drop it or rename the model
    robustness_df = robustness_df.drop(columns=["tokenizer_name"], errors="ignore")

    if output_format == "dataframe":
        return robustness_df
    if output_format == "markdown":
        return _build_markdown_table_str(
            robustness_df, higher_is_better=False, color_extremes=color_extremes
        )
    return _build_latex_table_str(
        robustness_df,
        latex_column_data,
        TITLE,
        "tab:robustness",
        higher_is_better=False,
        color_extremes=color_extremes,
    )


def get_accuracy_table_str(
    results_files_to_process: list[Path | str],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES,
    latex_column_data: list[list[str]] = LATEX_COLUMN_DATA,
    color_extremes: bool = True,
    output_format: OutputFormat = "latex",
) -> str | pd.DataFrame:
    """Generate formatted table for accuracy across given columns.

    column_data could be used for n-level headers e.g. [["Input", "<same>"], ["Non-EN", "EN"]] -> this will produce (Input combining two columns) // Non-EN & EN
    """
    print(
        f"Processing {len(results_files_to_process)} results files "
        "to compute robustness dataframe..."
    )
    accuracy_df = get_grouped_results_df(
        results_files_to_process, latex_table_categories
    )
    accuracy_df = accuracy_df.set_index("model_name").drop(
        columns=["tokenizer_name"], errors="ignore"
    )
    accuracy_df = accuracy_df.sort_values(by="Avg", ascending=False)

    if output_format == "dataframe":
        return accuracy_df
    if output_format == "markdown":
        return _build_markdown_table_str(
            accuracy_df, higher_is_better=True, color_extremes=color_extremes
        )
    caption = (
        "Accuracy under multilingual text perturbations. "
        "Values represent weighted-average $\\text{Acc}_{\\text{pert}}$ per category."
    )
    return _build_latex_table_str(
        accuracy_df,
        latex_column_data,
        caption,
        "tab:accuracy",
        higher_is_better=True,
        color_extremes=color_extremes,
    )


def get_canonical_performance_table_str(
    results_files_to_process: list[Path | str],
    latex_table_categories: dict[str, list[str]] = LATEX_TABLE_CATEGORIES_CANONICAL,
    latex_column_data: list[list[str]] | None = None,
    color_extremes: bool = True,
    output_format: OutputFormat = "latex",
) -> str | pd.DataFrame:
    """Generate formatted table for canonical accuracy.

    column_data could be used for n-level headers e.g. [["Input", "<same>"], ["Non-EN", "EN"]] -> this will produce (Input combining two columns) // Non-EN & EN
    """
    logging.info(
        f"Processing {len(results_files_to_process)} results files "
        "to compute robustness dataframe..."
    )
    canonical_df = _get_canonical_performance_df(
        results_files_to_process, latex_table_categories
    )
    canonical_df = canonical_df.set_index("model_name").drop(
        columns=["tokenizer_name"], errors="ignore"
    )
    canonical_df = canonical_df.sort_values(by="Avg", ascending=False)

    if output_format == "dataframe":
        return canonical_df
    if output_format == "markdown":
        return _build_markdown_table_str(
            canonical_df, higher_is_better=True, color_extremes=color_extremes
        )
    caption = (
        "Canonical (unperturbed) accuracy per category. "
        "Values represent mean $\\text{Acc}_{\\text{can}}$ "
        "across canonical tasks in each group."
    )
    if latex_column_data is None:
        latex_column_data = list(latex_table_categories.keys())

    return _build_latex_table_str(
        canonical_df,
        latex_column_data,
        caption,
        "tab:canonical_accuracy",
        higher_is_better=True,
        color_extremes=color_extremes,
    )
