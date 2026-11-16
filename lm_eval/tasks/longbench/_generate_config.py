# MIT License
#
# Copyright (c) 2023 THU-KEG & Zhipu AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse

from jinja2 import Environment


dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {question}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{question}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{question}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {question}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{question}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n{question}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n{question}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{question}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is:",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{question}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is:',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{question}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:",
    "repobench-p": "Please complete the code given below. \n{context}{question}Next line of code:",
}

dataset2metric = {
    "narrativeqa": "qa_f1_score",
    "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score",
    "multifieldqa_zh": "qa_f1_zh_score",
    "hotpotqa": "qa_f1_score",
    "2wikimqa": "qa_f1_score",
    "musique": "qa_f1_score",
    "dureader": "rouge_zh_score",
    "gov_report": "rouge_score",
    "qmsum": "rouge_score",
    "multi_news": "rouge_score",
    "vcsum": "rouge_zh_score",
    "trec": "classification_score",
    "triviaqa": "qa_f1_score",
    "samsum": "rouge_score",
    "lsht": "classification_score",
    "passage_retrieval_en": "retrieval_score",
    "passage_count": "count_score",
    "passage_retrieval_zh": "retrieval_zh_score",
    "lcc": "code_sim_score",
    "repobench-p": "code_sim_score",
}

DATASETS = [
    "2wikimqa",
    "2wikimqa_e",
    "dureader",
    "gov_report",
    "gov_report_e",
    "hotpotqa",
    "hotpotqa_e",
    "lcc",
    "lcc_e",
    "lsht",
    "multi_news",
    "multi_news_e",
    "multifieldqa_en",
    "multifieldqa_en_e",
    "multifieldqa_zh",
    "musique",
    "narrativeqa",
    "passage_count",
    "passage_count_e",
    "passage_retrieval_en",
    "passage_retrieval_en_e",
    "passage_retrieval_zh",
    "qasper",
    "qasper_e",
    "qmsum",
    "repobench-p",
    "repobench-p_e",
    "samsum",
    "samsum_e",
    "trec",
    "trec_e",
    "triviaqa",
    "triviaqa_e",
    "vcsum",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_prefix_path", default="")
    return parser.parse_args()


template_str = """
tag:
  - {{ tag[0] }}
task: {{ task }}
dataset_path: {{ dataset_path }}
test_split: {{ test_split }}
dataset_name: {{ dataset_name }}
doc_to_text: "{{ doc_to_text }}"
doc_to_target: '{{ doc_to_target }}'
process_results: {{ process_results }}
generation_kwargs:
  max_gen_toks: {{ generation_kwargs.max_gen_toks }}
  temperature: {{ generation_kwargs.temperature }}
  do_sample: {{ generation_kwargs.do_sample }}
  until: {% if has_newline %}["\\n"]{% else %}[]{% endif %}
metric_list:
  - metric: {{ metric_list[0].metric }}
    aggregation: {{ metric_list[0].aggregation }}
    higher_is_better: {{ metric_list[0].higher_is_better }}
metadata:
  version: {{ metadata.version }}
"""


if __name__ == "__main__":
    args = parse_args()
    env = Environment()
    template = env.from_string(template_str)
    for ds in DATASETS:
        df = ds[:-2] if ds.endswith("_e") else ds
        # from https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py#L52C25-L52C29

        # Now we just set a boolean flag to indicate whether we need a newline
        has_newline = df in ["trec", "triviaqa", "samsum", "lsht"]

        generation_kwargs = {
            "max_gen_toks": dataset2maxlen[df],
            "temperature": 1,
            "do_sample": False,
            # We'll handle the until value directly in the template
        }

        raw_doc_to_text = (
            dataset2prompt[df]
            .replace("\n", "\\n")
            .replace('"', '\\"')
            .replace("{", "{{")
            .replace("}", "}}")
        )
        metric_list = [
            {
                "metric": f'"{dataset2metric[df]}"',
                "aggregation": "mean",
                "higher_is_better": True,
            }
        ]

        data = {
            "tag": ["longbench_e" if ds.endswith("_e") else "longbench"],
            "task": f"longbench_{ds}",
            "dataset_path": "Xnhyacinth/LongBench",
            "test_split": "test",
            "dataset_name": ds,
            "doc_to_text": raw_doc_to_text,
            "doc_to_target": "{{answers}}",
            "process_results": f"!function metrics.get_{dataset2metric[df]}",
            "generation_kwargs": generation_kwargs,
            "has_newline": has_newline,  # Add the flag to the template context
            "metric_list": metric_list,
            "metadata": {"version": "5.0"},
        }

        # Render template
        rendered_yaml = template.render(**data)

        # Save to file
        with open(args.save_prefix_path + f"{ds}.yaml", "w") as f:
            f.write(rendered_yaml)
