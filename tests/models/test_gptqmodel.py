from typing import List

import pytest

import lm_eval


def assert_less_than(value, threshold, desc):
    if value is not None:
        assert float(value) < threshold, f"{desc} should be less than {threshold}"


@pytest.mark.skip(reason="requires CUDA")
class Test_GPTQModel:
    gptqmodel = pytest.importorskip("gptqmodel", minversion="1.0.9")
    MODEL_ID = "ModelCloud/Opt-125-GPTQ-4bit-10-25-2024"

    def test_gptqmodel(self) -> None:
        acc = "acc"
        acc_norm = "acc_norm"
        acc_value = None
        acc_norm_value = None
        task = "arc_easy"

        model_args = f"pretrained={self.MODEL_ID},gptqmodel=True"

        tasks: List[str] = [task]

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            device="cuda",
        )

        column = "results"
        dic = results.get(column, {}).get(self.task)
        if dic is not None:
            if "alias" in dic:
                _ = dic.pop("alias")
            items = sorted(dic.items())
            for k, v in items:
                m, _, f = k.partition(",")
                if m.endswith("_stderr"):
                    continue

                if m == acc:
                    acc_value = "%.4f" % v if isinstance(v, float) else v

                if m == acc_norm:
                    acc_norm_value = "%.4f" % v if isinstance(v, float) else v

            assert_less_than(acc_value, 0.43, "acc")
            assert_less_than(acc_norm_value, 0.39, "acc_norm")
