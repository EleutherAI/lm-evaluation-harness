import json
import sys

from pytablewriter import MarkdownTableWriter

json_file = json.load(open(sys.argv[1]))


results = []

for r in json_file["results"]:
    metric = [k[:-7] for k in r.keys() if "_stderr" in k][0]
    results.append(
        [
            r["prompt_name"],
            metric,
            "{0:.5g}".format(r[metric]),
            "{0:.5g}".format(r[metric + "_stderr"]),
        ]
    )


writer = MarkdownTableWriter(
    table_name=json_file["results"][0][
        "task_name"
    ],  # json_file["config"]["model"] + " on " + json_file["results"][0]["task_name"],
    headers=["prompt_name", "metric", "mean", "stderr"],
    value_matrix=results,
    margin=3,
)
writer.write_table()
