"""
Usage:
   python make_table_tasks.py --output <markdown_filename>
"""

import json
import logging
import os

from pytablewriter import LatexTableWriter, MarkdownTableWriter


logger = logging.getLogger(__name__)


def make_table(result_dict):
    """Generate table of results."""
    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in sorted(result_dict["results"].items()):
        version = result_dict["versions"][k]
        percent = k == "squad2"
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                if percent or m == "ppl":
                    values.append([k, version, m, "%.2f" % v, "±", "%.2f" % se])
                else:
                    values.append(
                        [k, version, m, "%.2f" % (v * 100), "±", "%.2f" % (se * 100)]
                    )
            else:
                if percent or m == "ppl":
                    values.append([k, version, m, "%.2f" % v, "", ""])
                else:
                    values.append([k, version, m, "%.2f" % (v * 100), "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


if __name__ == "__main__":
    # loop dirs and subdirs in results dir
    # for each dir, load json files
    for dirpath, dirnames, filenames in os.walk("../results"):
        # skip dirs without files
        if not filenames:
            continue
        path_readme = os.path.join(dirpath, "README.md")
        with open(path_readme, "w", encoding="utf-8") as f:
            # get path name, only last folder
            path_name = dirpath.split("/")[-1]
            f.write(f"# {path_name} \n\n")
        for filename in sorted([f for f in filenames if f.endswith(".json")]):
            path = os.path.join(dirpath, filename)
            with open(path, "r", encoding="utf-8") as f:
                result_dict = json.load(f)
            with open(path_readme, "a", encoding="utf-8") as f:
                f.write(f"## {filename} \n")
                f.write(f"{make_table(result_dict)} \n")
