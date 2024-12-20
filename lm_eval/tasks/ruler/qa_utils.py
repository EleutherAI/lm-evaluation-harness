# import random
# from collections import defaultdict
#
# from datasets import tqdm
#
#
# # def process_qa_docs(df: datasets.Dataset) -> datasets.Dataset:
# #     df = df.rename_column("question", "query")
# #     df = df.rename_column("answers", "outputs")
# #     df = df.map(lambda x: {"outputs": x["outputs"].get("text", [])})
# #     return df
#
#
# def process_qa_docs(dataset):
#     contexts = sorted(list(set(dataset["context"])))
#     ctx_to_id = {c: i for i, c in enumerate(contexts)}
#
#     title_contexts = defaultdict(list)
#     for ex in dataset:
#         title_contexts[ex["title"]].append(ctx_to_id[ex["context"]])
#
#     # Process function for map
#     def process_example(example):
#         if len(example["answers"]["text"]) == 0:
#             return None
#
#         ctx_id = ctx_to_id[example["context"]]
#         return {
#             "query": example["question"],
#             "outputs": example["answers"]["text"],
#             "context": [ctx_id],
#             "more_context": [
#                 i for i in title_contexts[example["title"]] if i != ctx_id
#             ],
#         }
#
#     return dataset.map(
#         process_example,
#         remove_columns=dataset.column_names,
#         filter_fn=lambda x: x is not None,
#     )
#
#
# def generate_input_output(index, num_docs, dataset, random_seed=42):
#     # Get current example
#     example = dataset[index]
#     curr_q = example["query"]
#     curr_a = example["outputs"]
#     curr_docs = example["context"]
#     curr_more = example["more_context"]
#     all_contexts = example["all_contexts"]
#
#     if num_docs < len(all_contexts):
#         if (num_docs - len(curr_docs)) > len(curr_more):
#             # Need additional random docs
#             addition_docs = [
#                 i for i in range(len(all_contexts)) if i not in curr_docs + curr_more
#             ]
#             all_docs = (
#                 curr_docs
#                 + curr_more
#                 + random.sample(
#                     addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
#                 )
#             )
#         else:
#             # Can just use some of the more_context
#             all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
#
#         all_docs = [all_contexts[idx] for idx in all_docs]
#     else:
#         all_docs = all_contexts
#
#     random.Random(random_seed).shuffle(all_docs)
#
#     # Assuming DOCUMENT_PROMPT is something like "Document {i}: {document}"
#     DOCUMENT_PROMPT = "Document {i}: {document}"
#     context = "\n\n".join(
#         [DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
#     )
#
#     # Assuming template is something like "{context}\nQuestion: {query}"
#     template = "{context}\nQuestion: {query}"
#     input_text = template.format(context=context, query=curr_q)
#
#     return input_text, curr_a
#
#
# def get_total_docs(example):
#     return len(example["all_contexts"]) if "all_contexts" in example else 0
#
#
# def generate_samples(
#     num_samples: int, max_seq_length: int, save_dir: str, incremental: int = 10
# ):
#     write_jsons = []
#     tokens_to_generate = tokens_to_generate
#
#     # Find the perfect num_docs
#     num_docs = incremental
#
#     total_tokens = 0  # Track the total tokens generated for this example
#     while total_tokens + tokens_to_generate < max_seq_length:
#         input_text, answer = generate_input_output(0, num_docs)
#         # Calculate the number of tokens in the example
#         total_tokens = len(TOKENIZER(input_text + f" {answer}").input_ids)
#         print(
#             f"Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}"
#         )
#         if total_tokens + tokens_to_generate > max_seq_length:
#             num_docs -= incremental
#             break
#
#         num_docs += incremental
#         if num_docs > len(DOCS):
#             num_docs = len(DOCS)
#             break
#     print("Number of documents:", num_docs)
#
#     # Generate samples
#     for index in tqdm(range(num_samples)):
#         used_docs = num_docs
#         while True:
#             try:
#                 input_text, answer = generate_input_output(
#                     index + pre_samples, used_docs
#                 )
#                 length = len(TOKENIZER(input_text).input_ids) + tokens_to_generate
#                 assert length <= max_seq_length, f"{length} exceeds max_seq_length."
#                 break
#             except:
#                 if used_docs > incremental:
#                     used_docs -= incremental
#
#         if remove_newline_tab:
#             input_text = " ".join(
#                 input_text.replace("\n", " ").replace("\t", " ").strip().split()
#             )
#
#         formatted_output = {
#             "index": index,
#             "input": input_text,
#             "outputs": answer,
#             "length": length,
#         }
#         write_jsons.append(formatted_output)
#
#     return write_jsons
