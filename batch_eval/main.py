import asyncio
import collections
import csv
import os
import time

import click
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@click.command()
@click.argument("datadir", required=True)
def main(datadir):
    loop = asyncio.get_event_loop()
    worker = InferenceWorker(loop)
    worker.load_model()
    worker_task = loop.create_task(worker.run())
    loop.run_until_complete(async_main(worker, datadir))
    worker_task.cancel()
    loop._run_once()  # https://stackoverflow.com/a/62443715
    loop.close()


async def async_main(worker, datadir):
    with open(
        os.path.join(datadir, "cloze_test_test__spring2016 - cloze_test_ALL_test.csv")
    ) as f:
        storycloze_test_examples = list(csv.DictReader(f))

    start_time = time.time()
    example_evaluations = await asyncio.gather(
        *(
            # `ensure_future` is needed to prevent `asyncio.gather` from returning the results out of order
            # See https://github.com/python/asyncio/pull/433
            asyncio.ensure_future(evaluate_example(worker, example))
            for example in storycloze_test_examples
        )
    )
    end_time = time.time()
    print(
        f"Total time for {len(storycloze_test_examples)} examples: {end_time - start_time}"
    )
    fraction_correct = len(
        [
            evaluation
            for evaluation in example_evaluations
            if evaluation["was_model_correct"]
        ]
    ) / float(len(example_evaluations))
    print(f"Fraction correct: {fraction_correct}")


async def evaluate_example(worker, example):
    storycloze_prompt = "{} {} {} {}".format(
        example["InputSentence1"],
        example["InputSentence2"],
        example["InputSentence3"],
        example["InputSentence4"],
    )

    # Calculate *per-token* likelihoods, as the paper did
    per_token_logits = await asyncio.gather(
        # `ensure_future` is needed to prevent `asyncio.gather` from returning the results out of order
        # See https://github.com/python/asyncio/pull/433
        asyncio.ensure_future(
            worker.compute_logit_per_token(
                storycloze_prompt + " " + example["RandomFifthSentenceQuiz1"]
            )
        ),
        asyncio.ensure_future(
            worker.compute_logit_per_token(
                storycloze_prompt + " " + example["RandomFifthSentenceQuiz2"]
            )
        ),
    )
    per_token_logit_for_sentence1 = per_token_logits[0].result()
    per_token_logit_for_sentence2 = per_token_logits[1].result()

    if per_token_logit_for_sentence1 > per_token_logit_for_sentence2:
        model_answer = example["RandomFifthSentenceQuiz1"]
        model_answer_code = "1"
    else:
        model_answer = example["RandomFifthSentenceQuiz2"]
        model_answer_code = "2"

    return {
        "model_answer": model_answer,
        "was_model_correct": model_answer_code == example["AnswerRightEnding"],
    }


InferenceRequest = collections.namedtuple("InferenceRequest", ["future", "input_text"])


class InferenceWorker:
    def __init__(self, loop):
        self.loop = loop
        self.inference_requests = []

        self.model = None
        self.tokenizer = None

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            # 117M
            pretrained_model_name_or_path="gpt2",
            config=AutoConfig.from_pretrained(
                "gpt2",
                # <|endoftext|>
                pad_token_id=50256,
            ),
        ).to("cuda")
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        prompt = "The quick brown fox jumps over"
        encoded_prompt = self.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")

        # Sanity check the model
        [output_token_ids] = self.model.generate(
            input_ids=encoded_prompt,
            max_length=100,
            tempareture=0,
            do_sample=False,
            num_return_sequences=1,
        )
        decoded_output = self.tokenizer.decode(output_token_ids.tolist())
        # Next word should be "the" ("The quick brown fox jumps over *the*...")
        assert decoded_output[len(prompt + " ") :].startswith("the")

    async def compute_logit_per_token(self, input_text):
        future = self.loop.create_future()
        self.inference_requests.append(
            InferenceRequest(future=future, input_text=input_text)
        )
        return future

    async def run(self):
        while True:
            if self.inference_requests:
                self.process_batch()
            if not self.inference_requests:
                # Need to sleep here, or else we'll get into an infinite loop.
                # The exact timeout duration doesn't matter much, as long as it's short.
                # It just needs to be not too much longer than the latency of one model inference, or ~10ms.
                await asyncio.sleep(0.01)

    def process_batch(self):
        # TODO: optimize batch size
        requests_to_process = self.inference_requests[:10]
        self.inference_requests = self.inference_requests[10:]

        for inference_request in requests_to_process:
            encoded_prompt_with_completion = self.tokenizer.encode(
                inference_request.input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            start_time = time.time()
            # This blocks the event loop, which is normally not recommended. (See https://docs.python.org/3/library/asyncio-dev.html#running-blocking-code)
            # But when we evaluate a model, we are running a big batch of evaluations and we don't care about responsiveness, only about how long it takes overall.
            # If we really want this to be non-blocking, we can move model inference to a separate thread.
            output_logits = self.model(encoded_prompt_with_completion).logits
            end_time = time.time()
            print(f"Time to evaluate once: {end_time - start_time}")

            # Align the output logits to the input tokens.
            logits_for_input_positions = output_logits[
                0,
                # The last logit needs to be dropped, because it's predicting the "next token", and it doesn't correspond to any input token
                :-1,
                :,
            ]
            input_tokens_at_positions_with_logits = encoded_prompt_with_completion[
                0,
                # The model does not predict the first input token, so the first token needs to be dropped.
                1:,
            ]
            # At each position, the model outputs ~50k logits, one for every possible token.
            # To get the logits of the tokens that were actually provided, we need to select the right logit at each position.
            logits_for_provided_tokens = torch.gather(
                logits_for_input_positions,
                1,
                input_tokens_at_positions_with_logits.unsqueeze(1),
            ).squeeze(1)

            inference_request.future.set_result(
                logits_for_provided_tokens.mean().item()
            )


if __name__ == "__main__":
    main()
