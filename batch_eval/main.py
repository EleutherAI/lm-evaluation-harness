import csv
import os
import time

import click
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@click.command()
@click.argument("datadir", required=True)
def main(datadir):
    model_runner = ModelRunner.create()

    with open(
        os.path.join(datadir, "cloze_test_test__spring2016 - cloze_test_ALL_test.csv")
    ) as f:
        storycloze_test_examples = list(csv.DictReader(f))

    start_time = time.time()
    example_evaluations = evaluate_examples(model_runner, storycloze_test_examples)
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


def evaluate_examples(model_runner, examples):
    prompts = [
        "{} {} {} {}".format(
            example["InputSentence1"],
            example["InputSentence2"],
            example["InputSentence3"],
            example["InputSentence4"],
        )
        for example in examples
    ]

    inputs_for_sentence_1 = [
        prompt + " " + example["RandomFifthSentenceQuiz1"]
        for prompt, example in zip(prompts, examples)
    ]
    inputs_for_sentence_2 = [
        prompt + " " + example["RandomFifthSentenceQuiz2"]
        for prompt, example in zip(prompts, examples)
    ]

    average_token_logits_with_sentence_1 = (
        model_runner.compute_average_token_logits_on_batch(inputs_for_sentence_1)
    )
    average_token_logits_with_sentence_2 = (
        model_runner.compute_average_token_logits_on_batch(inputs_for_sentence_2)
    )

    evaluation_results = []
    for i in range(len(examples)):
        if (
            average_token_logits_with_sentence_1[i]
            > average_token_logits_with_sentence_2[i]
        ):
            model_answer = examples[i]["RandomFifthSentenceQuiz1"]
            model_answer_code = "1"
        else:
            model_answer = examples[i]["RandomFifthSentenceQuiz2"]
            model_answer_code = "2"

        evaluation_results.append(
            {
                "model_answer": model_answer,
                "was_model_correct": model_answer_code
                == examples[i]["AnswerRightEnding"],
            }
        )
    return evaluation_results


class ModelRunner:
    def __init__(self):
        self.inference_requests = []
        self.num_inferences = 0

        self.model = None
        self.tokenizer = None

    @classmethod
    def create(cls):
        model_runner = cls()

        model_runner.model = AutoModelForCausalLM.from_pretrained(
            # 117M
            pretrained_model_name_or_path="gpt2",
            config=AutoConfig.from_pretrained(
                "gpt2",
                # <|endoftext|>
                pad_token_id=50256,
            ),
        ).to("cuda")
        model_runner.model = model_runner.model.eval()
        model_runner.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model_runner.tokenizer.pad_token = "<|endoftext|>"

        prompt = "The quick brown fox jumps over"
        encoded_prompt = model_runner.tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")

        # Sanity check the model
        [output_token_ids] = model_runner.model.generate(
            input_ids=encoded_prompt,
            max_length=100,
            tempareture=0,
            do_sample=False,
            num_return_sequences=1,
        )
        decoded_output = model_runner.tokenizer.decode(output_token_ids.tolist())
        # Next word should be "the" ("The quick brown fox jumps over *the*...")
        assert decoded_output[len(prompt + " ") :].startswith("the")

        return model_runner

    def compute_average_token_logits_on_batch(self, input_texts):
        """
        For each input text in the batch, compute the average logit (log-likelihood) over all tokens.

        For example, if an input sequence is 3 tokens long, and the token logits are [-1, -2, -3], the "average token logit" is -2.
        """
        # The ModelRunner can take a big batch on input_texts, and it can be as large as the caller wants.
        # But to prevent the GPU from running out of memory, we need to subdivide the overall batch
        # into "GPU batches", and the "GPU batch size" depends on the model and hardware.
        # For GPT-2-117M, a GPU can process a batch of roughly 10 or so inputs before the inference latency starts to increase.
        gpu_batch_size = 20

        average_token_logits = []
        for i in range(0, len(input_texts), gpu_batch_size):
            average_token_logits.extend(
                self._average_token_logits_on_gpu_batch(
                    input_texts[i : i + gpu_batch_size]
                )
            )
        return average_token_logits

    def _average_token_logits_on_gpu_batch(self, input_texts):
        tokenized_inputs = self.tokenizer(
            input_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )[
            # https://github.com/huggingface/transformers/issues/5480#issuecomment-653259416
            "input_ids"
        ].to(
            "cuda"
        )

        start_time = time.time()
        output_logits = self.model(tokenized_inputs).logits
        self.num_inferences += 1

        # Align the output logits to the input tokens.
        logits_for_input_positions = output_logits[
            # The batch dimension
            :,
            # The position dimension
            # The last logit needs to be dropped, because it's predicting the "next token", and it doesn't correspond to any input token
            :-1,
            # The embedding dimension
            :,
        ]
        input_tokens_at_positions_with_logits = tokenized_inputs[
            # The batch dimension
            :,
            # The position dimension
            # The model does not predict the first input token, so the first token needs to be dropped.
            1:,
        ]
        # At each position, the model outputs ~50k logits, one for every possible token.
        # To get the logits of the tokens that were actually provided, we need to select the right logit at each position.
        logits_for_provided_tokens = torch.gather(
            logits_for_input_positions,
            2,
            input_tokens_at_positions_with_logits.unsqueeze(2),
        ).squeeze(2)

        mask_for_non_padded_positions = input_tokens_at_positions_with_logits != 50256
        average_token_logits = (
            logits_for_provided_tokens * mask_for_non_padded_positions
        ).sum(1) / mask_for_non_padded_positions.sum(1)
        average_token_logits = average_token_logits.tolist()

        end_time = time.time()
        print(
            f"Time to evaluate once (inference #{self.num_inferences}): {end_time - start_time}"
        )
        return average_token_logits


if __name__ == "__main__":
    main()
