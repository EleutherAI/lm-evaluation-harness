import csv
import os

import click
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@click.command()
@click.argument("datadir", required=True)
def main(datadir):
    model = AutoModelForCausalLM.from_pretrained(
        # 117M
        pretrained_model_name_or_path="gpt2",
        config=AutoConfig.from_pretrained(
            "gpt2",
            # <|endoftext|>
            pad_token_id=50256,
        ),
    ).to("cuda")
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "The quick brown fox jumps over"
    encoded_prompt = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to("cuda")

    # Sanity check the model
    [output_token_ids] = model.generate(
        input_ids=encoded_prompt,
        max_length=100,
        tempareture=0,
        do_sample=False,
        num_return_sequences=1,
    )
    decoded_output = tokenizer.decode(output_token_ids.tolist())
    # Next word should be "the" ("The quick brown fox jumps over *the*...")
    print(decoded_output[len(prompt + " ") :][:10])
    assert decoded_output[len(prompt + " ") :].startswith("the")

    with open(
        os.path.join(datadir, "cloze_test_test__spring2016 - cloze_test_ALL_test.csv")
    ) as f:
        storycloze_test_examples = list(csv.DictReader(f))

    example_evaluations = [
        evaluate_example(model, tokenizer, example)
        for example in storycloze_test_examples
    ]
    fraction_correct = len(
        [
            evaluation
            for evaluation in example_evaluations
            if evaluation["was_model_correct"]
        ]
    ) / float(len(example_evaluations))
    print(f"Fraction correct: {fraction_correct}")


def evaluate_example(model, tokenizer, example):
    storycloze_prompt = "{} {} {} {}".format(
        example["InputSentence1"],
        example["InputSentence2"],
        example["InputSentence3"],
        example["InputSentence4"],
    )

    # Calculate *per-token* likelihoods, as the paper did
    per_token_logit_for_sentence1 = compute_per_token_logit_for_completion(
        model, tokenizer, storycloze_prompt, example["RandomFifthSentenceQuiz1"]
    )
    per_token_logit_for_sentence2 = compute_per_token_logit_for_completion(
        model, tokenizer, storycloze_prompt, example["RandomFifthSentenceQuiz2"]
    )

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


def compute_per_token_logit_for_completion(model, tokenizer, prompt, completion):
    encoded_prompt_with_completion = tokenizer.encode(
        prompt + " " + completion,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    output_logits = model(encoded_prompt_with_completion).logits

    # Align the output logits to the input tokens.
    # The last logit needs to be dropped, because it's predicting the "next token", and it doesn't correspond to any input token
    logits_for_input_positions = output_logits[0, :-1, :]
    # The model does not predict the first input token, so it needs to be dropped as well.
    input_tokens_at_positions_with_logits = encoded_prompt_with_completion[0, 1:]
    # At each position, the model outputs ~50k logits, one for every possible token.
    # To get the logits of the tokens that were actually provided, we need to select the right logit at each position.
    logits_for_provided_tokens = torch.gather(
        logits_for_input_positions,
        1,
        input_tokens_at_positions_with_logits.unsqueeze(1),
    ).squeeze(1)

    return logits_for_provided_tokens.mean().item()


if __name__ == "__main__":
    main()