from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import datasets
import pytest

from lm_eval import evaluator
from lm_eval.api.group import Group
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.aptbench import utils as aptbench_utils
from tests.utils import parser as ci_yaml_parser


APTBENCH_TASKS_DIR = (
    Path(__file__).resolve().parents[1] / "lm_eval" / "tasks" / "aptbench"
)


def _aptbench_task_manager(metadata: dict[str, Any] | None = None) -> TaskManager:
    """Create a fast TaskManager scoped strictly to APTBench YAMLs."""
    return TaskManager(
        include_path=str(APTBENCH_TASKS_DIR),
        include_defaults=False,
        metadata=metadata,
    )


EXPECTED_TASK_TO_PAIR: dict[str, tuple[str, str]] = {
    "aptbench_env_setup_plan": ("env_setup", "plan"),
    "aptbench_env_setup_action": ("env_setup", "action"),
    "aptbench_env_setup_error": ("env_setup", "error"),
    "aptbench_issue_fix_locate": ("issue_fix", "locate"),
    "aptbench_issue_fix_fix_patch": ("issue_fix", "fix_patch"),
    "aptbench_issue_fix_plan": ("issue_fix", "plan"),
    "aptbench_issue_fix_action": ("issue_fix", "action"),
    "aptbench_issue_fix_test_patch": ("issue_fix", "test_patch"),
    "aptbench_deepresearch_plan_en": ("deepresearch", "plan_en"),
    "aptbench_deepresearch_plan_zh": ("deepresearch", "plan_zh"),
    "aptbench_deepresearch_summ_ans_en": ("deepresearch", "summ_ans_en"),
    "aptbench_deepresearch_summ_ans_zh": ("deepresearch", "summ_ans_zh"),
    "aptbench_deepresearch_openend_plan_en": ("deepresearch", "openend_plan_en"),
    "aptbench_deepresearch_openend_citation_en": (
        "deepresearch",
        "openend_citation_en",
    ),
    "aptbench_deepresearch_openend_citation_zh": (
        "deepresearch",
        "openend_citation_zh",
    ),
    "aptbench_deepresearch_openend_quality_en": ("deepresearch", "openend_quality_en"),
    "aptbench_deepresearch_openend_quality_zh": ("deepresearch", "openend_quality_zh"),
    "aptbench_tool_acebench_api_select": ("tool", "acebench_api_select"),
    "aptbench_tool_acebench_api_param": ("tool", "acebench_api_param"),
    "aptbench_tool_bfcl_v4_api_select": ("tool", "bfcl_v4_api_select"),
    "aptbench_tool_bfcl_v4_api_param": ("tool", "bfcl_v4_api_param"),
    "aptbench_agentic_math_planning_single": ("agentic_math", "planning_single"),
    "aptbench_agentic_math_feedback_tf": ("agentic_math", "feedback_tf"),
    "aptbench_agentic_math_action_cal": ("agentic_math", "action_cal"),
}

EXPECTED_GROUPS = {
    "aptbench",
    "aptbench_swe",
    "aptbench_env_setup",
    "aptbench_issue_fix",
    "aptbench_deepresearch",
    "aptbench_tool",
    "aptbench_agentic_math",
}

# Independently extracted golden specs for offline regression testing (covers all
# 5 domains, 7 extractors, and 3 judge modes without reading aptbench_utils.TASK_SPECS).
INDEPENDENT_GOLDEN_CASES: list[dict[str, Any]] = [
    {
        "task_name": "aptbench_env_setup_plan",
        "dataset": "env_setup",
        "task": "plan",
        "template": "env_setup_plan_3shot.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "max_gen_toks": 10,
        "doc": {
            "uuid": "g1",
            "repo_info": " repo ",
            "choices": " (A) x ",
            "answer": "G",
        },
        "raw_output": "G)",
        "expected_target": "G",
        "expected_metrics": {"score": 1.0, "acc": 1.0},
    },
    {
        "task_name": "aptbench_env_setup_action",
        "dataset": "env_setup",
        "task": "action",
        "template": "env_setup_action_3shot.txt",
        "prompt_fields": {
            "$EXE_PLAN$": "execution_plan",
            "$EXE_CMDS$": "executed_cmds",
        },
        "answer_key": "target_cmd",
        "max_gen_toks": 30,
        "doc": {
            "uuid": "g2",
            "execution_plan": " p ",
            "executed_cmds": " c ",
            "target_cmd": "cd SVBench",
        },
        "raw_output": "cd SVBench; ls\n```",
        "expected_target": "cd SVBench",
        "expected_metrics": {"score": 1.0, "exact_match": 1.0},
    },
    {
        "task_name": "aptbench_issue_fix_action",
        "dataset": "issue_fix",
        "task": "action",
        "template": "issue_fix_action_3shot.txt",
        "prompt_fields": {"$TRAJ$": "trajs"},
        "answer_key": "answer",
        "max_gen_toks": 10,
        "doc": {
            "uuid": "g3",
            "trajs": " t ",
            "answer": "cd /testbed && python reproduce.py",
        },
        "raw_output": "cd /testbed && python reproduce.py\n```",
        "expected_target": "cd /testbed && python reproduce.py",
        "expected_metrics": {"score": 1.0, "exact_match": 1.0},
    },
    {
        "task_name": "aptbench_deepresearch_summ_ans_en",
        "dataset": "deepresearch",
        "task": "summ_ans_en",
        "template": "deepresearch_summ_ans_en_3shot.txt",
        "prompt_fields": {"$QUERY$": "query", "$TRAJ$": "trajectory"},
        "answer_key": "answer",
        "max_gen_toks": 30,
        "doc": {
            "uuid": "g4",
            "query": " q ",
            "trajectory": " tr ",
            "answer": "Howard Atwood Kelly",
        },
        "raw_output": '"Howard Atwood Kelly"]\n',
        "expected_target": "Howard Atwood Kelly",
        "expected_metrics": {
            "score": 0.0,
            "exact_match": 0.0,
            "exact_match_flex": 1.0,
        },
    },
    {
        "task_name": "aptbench_deepresearch_openend_citation_zh",
        "dataset": "deepresearch",
        "task": "openend_citation_zh",
        "template": "deepresearch_openend_citation_zh_3shot.txt",
        "prompt_fields": {
            "$ARTICLE$": "article",
            "$CHOICES$": "choices",
            "$WEB_PAGE$": "url_content",
        },
        "answer_key": "answer",
        "max_gen_toks": 10,
        "doc": {
            "uuid": "g5",
            "article": " a ",
            "choices": " ch ",
            "url_content": " u ",
            "answer": ["A", "B", "E"],
        },
        "raw_output": "E,B,A)\n",
        "expected_target": "A,B,E",
        "expected_metrics": {"score": 1.0, "acc": 1.0},
    },
    {
        "task_name": "aptbench_tool_acebench_api_param",
        "dataset": "tool",
        "task": "acebench_api_param",
        "template": "tool_acebench_api_param_3shot.txt",
        "prompt_fields": {
            "$USER_PROMPT$": "user_prompt",
            "$FUNCS$": "function",
            "$ANSWER_SELECT$": "answer4select",
            "$PARA_NAME$": "param_name",
        },
        "answer_key": "answer4param",
        "max_gen_toks": 30,
        "doc": {
            "uuid": "g6",
            "user_prompt": " up ",
            "function": " fn ",
            "answer4select": " sel ",
            "param_name": " loc ",
            "answer4param": '{"latitude": 37.9838}',
        },
        "raw_output": '{"latitude": 37.9838}\n```',
        "expected_target": '{"latitude": 37.9838}',
        "expected_metrics": {"score": 1.0, "exact_match": 1.0},
    },
    {
        "task_name": "aptbench_agentic_math_planning_single",
        "dataset": "agentic_math",
        "task": "planning_single",
        "template": "math_planning_single_3shot.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "max_gen_toks": 10,
        "doc": {"uuid": "g7", "repo_info": " q ", "choices": " ch ", "answer": "E"},
        "raw_output": "*e*:",
        "expected_target": "E",
        "expected_metrics": {"score": 1.0, "acc": 1.0},
    },
    {
        "task_name": "aptbench_agentic_math_feedback_tf",
        "dataset": "agentic_math",
        "task": "feedback_tf",
        "template": "math_feedback_tf.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "max_gen_toks": 10,
        "doc": {
            "uuid": "g8",
            "repo_info": " sol ",
            "choices": "true; false",
            "answer": "wrong",
        },
        "raw_output": "**Wrong**##",
        "expected_target": "wrong",
        "expected_metrics": {"score": 1.0, "acc": 1.0},
    },
    {
        "task_name": "aptbench_agentic_math_action_cal",
        "dataset": "agentic_math",
        "task": "action_cal",
        "template": "math_action_calculation.txt",
        "prompt_fields": {"$REPO_INFO$": "repo_info", "$CHOICES$": "choices"},
        "answer_key": "answer",
        "max_gen_toks": 10,
        "doc": {
            "uuid": "g9",
            "repo_info": " 1+1 ",
            "choices": " (A) 2 ",
            "answer": "A",
        },
        "raw_output": "a ",
        "expected_target": "A",
        "expected_metrics": {"score": 1.0, "acc": 1.0},
    },
]

# SHA-256 prompt hashes computed independently from pinned commit
# 9aab1301ab2c743be6bf11d1a2f3c3c6bc123ffc for row 0 of representative tasks.
PINNED_COMMIT_GOLDEN_HASHES: dict[str, tuple[str, str]] = {
    "aptbench_env_setup_action": (
        "f8026b9d-bd5d-474c-82b0-4aeb7d65471b",
        "f37826dd140f19ade21870969cf72e7d825d63541f903fdcad95e608cdcf3c5c",
    ),
    "aptbench_issue_fix_action": (
        "cdcbf913-5bb1-40c9-8e46-8dae0aca5944",
        "39dbdb45994860c42cab5151048b56eb4f3fa9008ff7066275d3f23eb9f4fad3",
    ),
    "aptbench_deepresearch_openend_citation_zh": (
        "d52d9713-e5b3-4f21-8ae5-301d5a8b2c46",
        "3b9abe6fdaeb2da8186c53225b1438638a1866869c160dd1880009266a7888f1",
    ),
    "aptbench_deepresearch_summ_ans_en": (
        "28298055-db51-4c7d-8aab-b00759272494",
        "4dbacae1ddf19511b206c7107d632037ca27746223d5d59a39385016b7c3c33e",
    ),
    "aptbench_tool_acebench_api_param": (
        "1b10d236-0861-48d9-bc04-878256392114",
        "fe2ccfedcbbccd2e7619edfb5b0b09f1f5720db693db3ce5bea5f184223ebabf",
    ),
    "aptbench_agentic_math_action_cal": (
        "5b93b996-73d6-4ca2-84b6-364e97966dfc",
        "24b630c1472f26f84a7d7e9eac38c4001fc1acc994b379e67b3119793108cfdc",
    ),
}


def _get_local_aptbench_repo() -> Path | None:
    """Return local APTBench checkout path from APTBENCH_DIR when valid."""
    env_dir = os.environ.get("APTBENCH_DIR")
    if not env_dir:
        return None
    repo = Path(env_dir)
    if (repo / "code" / "predict.py").is_file() and (repo / "data").is_dir():
        return repo
    return None


def _build_dummy_aptbench_tree(root: Path, *, marker: str = "") -> None:
    """Create minimal offline APTBench data and prompt files for all 24 tasks."""
    for (dataset, task), spec in aptbench_utils.TASK_SPECS.items():
        prompt_path = root / "code" / "prompts" / spec["template"]
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        placeholders = " ".join(spec["prompt_fields"].keys())
        suffix = f" {marker}" if marker else ""
        prompt_path.write_text(
            f"Header {{{{raw_jinja}}}} {{% block %}} {placeholders}{suffix}\nAnswer: ",
            encoding="utf-8",
        )

        data_path = root / "data" / dataset / task / "input_data.jsonl"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {"uuid": f"{dataset}-{task}-0"}
        for col in spec["prompt_fields"].values():
            row[col] = f"  val_{col}  "
        if spec["judge"] == "cite_set":
            row[spec["answer_key"]] = ["A", "B"]
        elif (dataset, task) == ("agentic_math", "feedback_tf"):
            row[spec["answer_key"]] = "correct"
        else:
            row[spec["answer_key"]] = "A"
        data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _parse_upstream_predict_py(
    predict_py: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Independently parse template paths and TASK_CONFIG from predict.py via AST."""
    tree = ast.parse(predict_py.read_text(encoding="utf-8"))
    template_vars: dict[str, str] = {}
    parsed_configs: dict[tuple[str, str], dict[str, Any]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if (
            target.id.startswith("template_")
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            template_vars[target.id] = Path(node.value.args[0].value).name
        elif target.id == "TASK_CONFIG" and isinstance(node.value, ast.Dict):
            for key_node, val_node in zip(
                node.value.keys, node.value.values, strict=True
            ):
                if key_node is None or not isinstance(val_node, ast.Dict):
                    continue
                task_key = tuple(ast.literal_eval(key_node))
                entry: dict[str, Any] = {}
                for k_node, v_node in zip(val_node.keys, val_node.values, strict=True):
                    if not isinstance(k_node, ast.Constant):
                        continue
                    field_name = k_node.value
                    if field_name == "template" and isinstance(v_node, ast.Name):
                        entry["template"] = template_vars[v_node.id]
                    elif field_name == "extraction_func" and isinstance(
                        v_node, ast.Name
                    ):
                        entry["extraction_func"] = v_node.id
                    elif field_name in {
                        "prompt_fields",
                        "max_new_tokens",
                        "answer_key",
                    }:
                        entry[field_name] = ast.literal_eval(v_node)
                parsed_configs[task_key] = entry

    return parsed_configs


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A)", "A"),
        ("**B**\nexplanation", "B"),
        ("c) choice", "c"),
        ("answer is A", None),
        ("", None),
    ],
)
def test_extract_answer(response: str, expected: str | None) -> None:
    assert aptbench_utils.extract_answer(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("pip install -e .; pytest\n```", "pip install -e ."),
        ("cd repo && ls; echo done", "cd repo && ls; echo done"),
        ("conda activate env\nnext", "conda activate env"),
        ("", ""),
    ],
)
def test_extract_answer_action(response: str, expected: str) -> None:
    assert aptbench_utils.extract_answer_action(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Howard Atwood Kelly]\n", "Howard Atwood Kelly"),
        ("1994]", "1994"),
        ("missing bracket", None),
    ],
)
def test_extract_answer_summ_ans(response: str, expected: str | None) -> None:
    assert aptbench_utils.extract_answer_summ_ans(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A,B,E)\n", "A,B,E"),
        ("A)", "A"),
        ("A,B", None),
    ],
)
def test_extract_answer_cite(response: str, expected: str | None) -> None:
    assert aptbench_utils.extract_answer_cite(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("b)", "B"),
        ("*e*:", "E"),
        ("E", "E"),
        ("F)", None),
    ],
)
def test_extract_answer_choice_ae(response: str, expected: str | None) -> None:
    assert aptbench_utils.extract_answer_choice_ae(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("d)", "D"),
        ("a\n", "A"),
        ("c ", "C"),
        ("E)", None),
    ],
)
def test_extract_answer_choice_ad(response: str, expected: str | None) -> None:
    assert aptbench_utils.extract_answer_choice_ad(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("CORRECT##", "correct"),
        ("**Wrong**", "wrong"),
        ("unknown", None),
    ],
)
def test_extract_answer_true_false(response: str, expected: str | None) -> None:
    assert aptbench_utils.extract_answer_true_false(response) == expected


def test_score_response_all_judge_modes() -> None:
    mcq_spec = aptbench_utils.TASK_SPECS[("env_setup", "plan")]
    mcq_doc = {"answer": "G"}
    assert aptbench_utils.score_response(mcq_doc, ["G)"], mcq_spec) == {
        "score": 1.0,
        "acc": 1.0,
    }
    assert aptbench_utils.score_response(mcq_doc, [""], mcq_spec) == {
        "score": 0.0,
        "acc": 0.0,
    }

    tc_spec = aptbench_utils.TASK_SPECS[("env_setup", "action")]
    tc_doc = {"target_cmd": "conda activate ESAM"}
    assert aptbench_utils.score_response(
        tc_doc, ["conda activate ESAM\n```"], tc_spec
    ) == {"score": 1.0, "exact_match": 1.0}
    assert aptbench_utils.score_response(tc_doc, [""], tc_spec) == {
        "score": 0.0,
        "exact_match": 0.0,
    }

    cite_spec = aptbench_utils.TASK_SPECS[("deepresearch", "openend_citation_en")]
    cite_doc = {"answer": ["A", "B", "E"]}
    assert aptbench_utils.score_response(cite_doc, ["E,B,A)"], cite_spec) == {
        "score": 1.0,
        "acc": 1.0,
    }
    assert aptbench_utils.score_response(cite_doc, ["A,B)"], cite_spec) == {
        "score": 0.0,
        "acc": 0.0,
    }
    assert aptbench_utils.score_response(cite_doc, [""], cite_spec) == {
        "score": 0.0,
        "acc": 0.0,
    }

    summ_spec = aptbench_utils.TASK_SPECS[("deepresearch", "summ_ans_en")]
    summ_doc = {"answer": "Howard Atwood Kelly"}
    assert aptbench_utils.score_response(
        summ_doc, ["Howard Atwood Kelly]"], summ_spec
    ) == {"score": 1.0, "exact_match": 1.0, "exact_match_flex": 1.0}
    assert aptbench_utils.score_response(
        summ_doc, ['"Howard Atwood Kelly"]'], summ_spec
    ) == {"score": 0.0, "exact_match": 0.0, "exact_match_flex": 1.0}


def test_model_args_revision_does_not_override_aptbench_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generic model_args keys (revision/data_dir/prompt_dir) must never collide."""
    for env_key in (
        "APTBENCH_DIR",
        "APTBENCH_DATA_DIR",
        "APTBENCH_PROMPT_DIR",
        "APTBENCH_REVISION",
    ):
        monkeypatch.delenv(env_key, raising=False)

    aptbench_dir, data_dir, prompt_dir, revision = (
        aptbench_utils._resolve_source_options(
            {"revision": "step10000", "data_dir": "/bad/data", "prompt_dir": "/bad/p"}
        )
    )
    assert aptbench_dir is None
    assert data_dir is None
    assert prompt_dir is None
    assert revision == aptbench_utils.APTBENCH_COMMIT

    # End-to-end check without aptbench_dir: simple_evaluate(..., model_args="revision=step10000")
    # must merge model_args into metadata yet still hit the pinned APTBENCH_COMMIT remote URLs.
    tree = tmp_path / "repo"
    _build_dummy_aptbench_tree(tree)
    captured_urls: list[str] = []
    raw_prefix = f"https://raw.githubusercontent.com/{aptbench_utils.APTBENCH_REPO}/"

    def _fake_download(_self: Any, url: str) -> str:
        captured_urls.append(url)
        assert url.startswith(raw_prefix), f"Unexpected download URL: {url}"
        rev, rel_path = url[len(raw_prefix) :].split("/", maxsplit=1)
        assert rev == aptbench_utils.APTBENCH_COMMIT
        return str(tree / rel_path)

    monkeypatch.setattr(datasets.DownloadManager, "download", _fake_download)

    results = evaluator.simple_evaluate(
        model="dummy",
        model_args="revision=step10000",
        tasks=["aptbench_agentic_math_action_cal"],
        limit=1,
    )
    assert "aptbench_agentic_math_action_cal" in results["results"]
    assert len(captured_urls) == 2
    assert any(
        url.endswith(
            f"/{aptbench_utils.APTBENCH_COMMIT}/data/agentic_math/action_cal/input_data.jsonl"
        )
        for url in captured_urls
    )
    assert any(
        url.endswith(
            f"/{aptbench_utils.APTBENCH_COMMIT}/code/prompts/math_action_calculation.txt"
        )
        for url in captured_urls
    )
    assert all("step10000" not in url for url in captured_urls)

    # Also verify aptbench_data_dir, aptbench_prompt_dir, and env var overrides work.
    custom_data = tmp_path / "custom_data" / "deepresearch" / "openend_citation_en"
    custom_data.mkdir(parents=True)
    (custom_data / "input_data.jsonl").write_text(
        json.dumps(
            {
                "uuid": "u1",
                "article": " art ",
                "choices": " ch ",
                "url_content": " web ",
                "answer": ["A", "C", "E"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    custom_prompts = tmp_path / "custom_prompts"
    custom_prompts.mkdir(parents=True)
    (custom_prompts / "deepresearch_openend_citation_en_3shot.txt").write_text(
        "A=$ARTICLE$|C=$CHOICES$|W=$WEB_PAGE$(",
        encoding="utf-8",
    )

    monkeypatch.setenv("APTBENCH_DATA_DIR", str(tmp_path / "custom_data"))
    monkeypatch.setenv("APTBENCH_PROMPT_DIR", str(custom_prompts))
    tm = _aptbench_task_manager()
    task_obj = tm.load(["aptbench_deepresearch_openend_citation_en"])["tasks"][
        "aptbench_deepresearch_openend_citation_en"
    ]
    doc = next(iter(task_obj.test_docs()))
    assert task_obj.doc_to_text(doc) == "A=art|C=ch|W=web("
    assert task_obj.doc_to_target(doc) == "A,C,E"


def test_log_samples_no_path_leak_and_path_independent_hashes(
    tmp_path: Path,
) -> None:
    """Samples must not leak filesystem paths and must produce identical hashes."""
    tree_a = tmp_path / "dir_a"
    tree_b = tmp_path / "dir_b"
    _build_dummy_aptbench_tree(tree_a)
    _build_dummy_aptbench_tree(tree_b)

    res_a = evaluator.simple_evaluate(
        model="dummy",
        tasks=["aptbench_env_setup_plan"],
        task_manager=_aptbench_task_manager(metadata={"aptbench_dir": str(tree_a)}),
        limit=1,
        log_samples=True,
    )
    res_b = evaluator.simple_evaluate(
        model="dummy",
        tasks=["aptbench_env_setup_plan"],
        task_manager=_aptbench_task_manager(metadata={"aptbench_dir": str(tree_b)}),
        limit=1,
        log_samples=True,
    )

    sample_a = res_a["samples"]["aptbench_env_setup_plan"][0]
    sample_b = res_b["samples"]["aptbench_env_setup_plan"][0]

    assert not any(k.startswith("_aptbench_") for k in sample_a["doc"])
    serialized_doc = json.dumps(sample_a["doc"])
    assert "dir_a" not in serialized_doc
    assert str(tmp_path) not in serialized_doc

    assert sample_a["doc_hash"] == sample_b["doc_hash"]
    assert sample_a["prompt_hash"] == sample_b["prompt_hash"]
    assert sample_a["target_hash"] == sample_b["target_hash"]

    # Two distinct TaskManagers in the same process with different prompt templates
    # must stay completely isolated on their respective Task instances.
    prompt_dir_x = tmp_path / "prompts_x"
    prompt_dir_y = tmp_path / "prompts_y"
    prompt_dir_x.mkdir()
    prompt_dir_y.mkdir()
    (prompt_dir_x / "env_setup_plan_3shot.txt").write_text(
        "TEMPLATE_X: $REPO_INFO$ | $CHOICES$(", encoding="utf-8"
    )
    (prompt_dir_y / "env_setup_plan_3shot.txt").write_text(
        "TEMPLATE_Y: $REPO_INFO$ | $CHOICES$(", encoding="utf-8"
    )

    task_x = _aptbench_task_manager(
        metadata={"aptbench_dir": str(tree_a), "aptbench_prompt_dir": str(prompt_dir_x)}
    ).load(["aptbench_env_setup_plan"])["tasks"]["aptbench_env_setup_plan"]
    task_y = _aptbench_task_manager(
        metadata={"aptbench_dir": str(tree_a), "aptbench_prompt_dir": str(prompt_dir_y)}
    ).load(["aptbench_env_setup_plan"])["tasks"]["aptbench_env_setup_plan"]

    doc_x = next(iter(task_x.test_docs()))
    doc_y = next(iter(task_y.test_docs()))
    assert task_x.doc_to_text(doc_x).startswith("TEMPLATE_X:")
    assert task_y.doc_to_text(doc_y).startswith("TEMPLATE_Y:")


def test_local_prompt_edit_reloads_fresh_in_same_process(tmp_path: Path) -> None:
    """Editing a local prompt file must take effect immediately on reload."""
    tree = tmp_path / "editable"
    _build_dummy_aptbench_tree(tree, marker="VERSION_1")

    task_v1 = _aptbench_task_manager(metadata={"aptbench_dir": str(tree)}).load(
        ["aptbench_env_setup_plan"]
    )["tasks"]["aptbench_env_setup_plan"]
    assert "VERSION_1" in task_v1.doc_to_text(next(iter(task_v1.test_docs())))

    prompt_file = tree / "code" / "prompts" / "env_setup_plan_3shot.txt"
    prompt_file.write_text(
        prompt_file.read_text(encoding="utf-8").replace("VERSION_1", "VERSION_2"),
        encoding="utf-8",
    )

    task_v2 = _aptbench_task_manager(metadata={"aptbench_dir": str(tree)}).load(
        ["aptbench_env_setup_plan"]
    )["tasks"]["aptbench_env_setup_plan"]
    text_v2 = task_v2.doc_to_text(next(iter(task_v2.test_docs())))
    assert "VERSION_2" in text_v2
    assert "VERSION_1" not in text_v2


def test_missing_or_null_required_field_fails_fast(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing or null required fields must raise ValueError instead of dropping columns."""
    bad_dir = tmp_path / "data" / "env_setup" / "plan"
    bad_dir.mkdir(parents=True)
    jsonl_path = bad_dir / "input_data.jsonl"

    # Line 1 omits 'choices', line 2 is complete (previously triggered Dataset.from_list column drop).
    jsonl_path.write_text(
        json.dumps({"uuid": "bad-0", "repo_info": "r0", "answer": "A"})
        + "\n"
        + json.dumps(
            {"uuid": "ok-1", "repo_info": "r1", "choices": "(A) x", "answer": "A"}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"line 1.*bad-0.*choices"):
        aptbench_utils.load_aptbench_dataset(
            aptbench_dataset="env_setup",
            aptbench_task="plan",
            aptbench_data_dir=str(tmp_path / "data"),
        )

    # Null required field must also fail fast.
    jsonl_path.write_text(
        json.dumps(
            {"uuid": "null-0", "repo_info": None, "choices": "(A) x", "answer": "A"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"line 1.*null-0.*repo_info"):
        aptbench_utils.load_aptbench_dataset(
            aptbench_dataset="env_setup",
            aptbench_task="plan",
            aptbench_data_dir=str(tmp_path / "data"),
        )

    # Direct format_prompt call with missing field must log a warning.
    spec = aptbench_utils.TASK_SPECS[("env_setup", "plan")]
    with caplog.at_level(logging.WARNING, logger=aptbench_utils.eval_logger.name):
        rendered = aptbench_utils.format_prompt(
            "R=$REPO_INFO$ C=$CHOICES$(", {"uuid": "u0", "repo_info": "repo"}, spec
        )
    assert rendered == "R=repo C=$CHOICES$("
    assert any("missing required prompt field" in r.message for r in caplog.records)


def test_optional_scalar_metadata_preserved_and_download_error_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional scalar metadata columns survive cleanly and download errors give local guidance."""
    data_dir = tmp_path / "data" / "agentic_math" / "action_cal"
    data_dir.mkdir(parents=True)
    (data_dir / "input_data.jsonl").write_text(
        json.dumps(
            {
                "uuid": "m0",
                "repo_info": "q0",
                "choices": "c0",
                "answer": "A",
            }
        )
        + "\n"
        + json.dumps(
            {
                "uuid": "m1",
                "repo_info": "q1",
                "choices": "c1",
                "answer": "B",
                "ability": "algebra",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ds = aptbench_utils.load_aptbench_dataset(
        aptbench_dataset="agentic_math",
        aptbench_task="action_cal",
        aptbench_data_dir=str(tmp_path / "data"),
    )["test"]
    assert "ability" in ds.column_names
    assert ds[0]["ability"] == ""
    assert ds[1]["ability"] == "algebra"

    class _FailingDownloadManager:
        def download(self, url: str) -> str:
            raise ConnectionError("offline")

    monkeypatch.setattr(aptbench_utils, "_DOWNLOAD_MANAGER", _FailingDownloadManager())
    with pytest.raises(RuntimeError, match=r"APTBENCH_DIR"):
        aptbench_utils._download_remote_file(
            "code/prompts/math_action_calculation.txt", aptbench_utils.APTBENCH_COMMIT
        )


def test_task_manager_discovery_and_ci_yaml_parser() -> None:
    yaml_files = sorted(APTBENCH_TASKS_DIR.glob("*.yaml"))
    assert len(yaml_files) == 31
    rel_paths = [str(p.relative_to(APTBENCH_TASKS_DIR.parents[2])) for p in yaml_files]
    parsed_names = set(ci_yaml_parser(rel_paths))
    assert parsed_names == set(EXPECTED_TASK_TO_PAIR) | EXPECTED_GROUPS

    tm = TaskManager()
    for name in EXPECTED_TASK_TO_PAIR:
        assert name in tm.all_subtasks
    for group_name in EXPECTED_GROUPS:
        assert group_name in tm.all_groups

    for yaml_path in yaml_files:
        cfg = load_yaml(yaml_path)
        assert isinstance(cfg, dict)


def test_group_load_and_macro_aggregation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify real loaded nested Group objects flatten leaves cleanly."""
    _build_dummy_aptbench_tree(tmp_path)

    tm = _aptbench_task_manager(metadata={"aptbench_dir": str(tmp_path)})
    loaded = tm.load(["aptbench"])
    root_group = loaded["groups"]["aptbench"]
    assert isinstance(root_group, Group)
    assert set(loaded["groups"]) == EXPECTED_GROUPS

    leaf_tasks = root_group.get_all_tasks()
    assert len(leaf_tasks) == 24
    assert {t.task_name for t in leaf_tasks} == set(EXPECTED_TASK_TO_PAIR)

    tasks_by_name = {t.task_name: t for t in leaf_tasks}
    assert (
        tasks_by_name["aptbench_issue_fix_action"].config.generation_kwargs[
            "max_gen_toks"
        ]
        == 10
    )
    assert (
        tasks_by_name["aptbench_env_setup_action"].config.generation_kwargs[
            "max_gen_toks"
        ]
        == 30
    )
    assert (
        tasks_by_name["aptbench_deepresearch_summ_ans_en"].config.generation_kwargs[
            "max_gen_toks"
        ]
        == 30
    )

    # Distinct per-domain values and uneven sample_len verify unweighted leaf flattening
    # (e.g., aptbench_swe = 3/8 = 0.375 rather than (1.0 + 0.0)/2 = 0.5).
    domain_scores = {
        "aptbench_env_setup_": 1.0,
        "aptbench_issue_fix_": 0.0,
        "aptbench_deepresearch_": 0.5,
        "aptbench_tool_": 0.25,
        "aptbench_agentic_math_": 0.75,
    }
    task_metrics: dict[str, Any] = {}
    for idx, task_name in enumerate(sorted(EXPECTED_TASK_TO_PAIR)):
        val = next(
            score
            for prefix, score in domain_scores.items()
            if task_name.startswith(prefix)
        )
        task_metrics[task_name] = {
            "alias": task_name,
            "name": task_name,
            "sample_len": (idx + 1) * 100,
            "score,none": val,
            "score_stderr,none": 0.01,
        }

    with caplog.at_level(logging.WARNING, logger="lm_eval.api.group"):
        env_agg = loaded["groups"]["aptbench_env_setup"].aggregate(task_metrics)
        fix_agg = loaded["groups"]["aptbench_issue_fix"].aggregate(task_metrics)
        swe_agg = loaded["groups"]["aptbench_swe"].aggregate(task_metrics)
        dr_agg = loaded["groups"]["aptbench_deepresearch"].aggregate(task_metrics)
        tool_agg = loaded["groups"]["aptbench_tool"].aggregate(task_metrics)
        math_agg = loaded["groups"]["aptbench_agentic_math"].aggregate(task_metrics)
        root_agg = root_group.aggregate(task_metrics)

    assert not any("missing" in rec.message.lower() for rec in caplog.records)
    assert env_agg["score,none"] == pytest.approx(1.0)
    assert fix_agg["score,none"] == pytest.approx(0.0)
    assert swe_agg["score,none"] == pytest.approx(3 / 8)
    assert dr_agg["score,none"] == pytest.approx(0.5)
    assert tool_agg["score,none"] == pytest.approx(0.25)
    assert math_agg["score,none"] == pytest.approx(0.75)
    expected_root = (3 * 1.0 + 5 * 0.0 + 9 * 0.5 + 4 * 0.25 + 3 * 0.75) / 24
    assert root_agg["score,none"] == pytest.approx(expected_root)


def test_offline_golden_fixtures_independent(tmp_path: Path) -> None:
    """Verify prompt rendering, targets, max_gen_toks, and scoring using independent specs."""
    for case in INDEPENDENT_GOLDEN_CASES:
        prompt_path = tmp_path / "code" / "prompts" / case["template"]
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        placeholders = " | ".join(case["prompt_fields"].keys())
        raw_tpl = f"Golden {{{{literal}}}} {placeholders}\nOut:"
        prompt_path.write_text(raw_tpl, encoding="utf-8")

        data_path = (
            tmp_path / "data" / case["dataset"] / case["task"] / "input_data.jsonl"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(json.dumps(case["doc"]) + "\n", encoding="utf-8")

    tm = _aptbench_task_manager(metadata={"aptbench_dir": str(tmp_path)})
    loaded = tm.load([case["task_name"] for case in INDEPENDENT_GOLDEN_CASES])
    tasks_by_name = loaded["tasks"]

    for case in INDEPENDENT_GOLDEN_CASES:
        task_obj = tasks_by_name[case["task_name"]]
        assert task_obj.config.generation_kwargs["max_gen_toks"] == case["max_gen_toks"]

        docs = list(task_obj.test_docs())
        assert len(docs) == 1
        doc = docs[0]
        assert not any(k.startswith("_aptbench_") for k in doc)

        expected_prompt = (
            "Golden {{literal}} " + " | ".join(case["prompt_fields"].keys()) + "\nOut:"
        )
        for placeholder, col in case["prompt_fields"].items():
            expected_prompt = expected_prompt.replace(
                placeholder, str(case["doc"][col]).strip()
            )
        assert task_obj.doc_to_text(doc) == expected_prompt
        assert task_obj.doc_to_target(doc) == case["expected_target"]
        assert (
            task_obj.process_results(doc, [case["raw_output"]])
            == case["expected_metrics"]
        )


def test_parity_with_local_aptbench_repo() -> None:
    """Verify prompt text, targets, and config against an APTBENCH_DIR checkout via AST."""
    repo = _get_local_aptbench_repo()
    if repo is None:
        pytest.skip(
            "Set APTBENCH_DIR to a local APTBench checkout to run parity tests."
        )

    upstream_configs = _parse_upstream_predict_py(repo / "code" / "predict.py")
    assert len(upstream_configs) == 24

    tm = _aptbench_task_manager(metadata={"aptbench_dir": str(repo)})
    loaded = tm.load(["aptbench"])
    tasks_by_name = {
        t.task_name: t for t in loaded["groups"]["aptbench"].get_all_tasks()
    }

    for task_name, (dataset, task) in EXPECTED_TASK_TO_PAIR.items():
        upstream = upstream_configs[(dataset, task)]
        assert (
            aptbench_utils.TASK_SPECS[(dataset, task)]["extractor"].__name__
            == upstream["extraction_func"].lower()
        )
        raw_template = (repo / "code" / "prompts" / upstream["template"]).read_text(
            encoding="utf-8"
        )
        data_path = repo / "data" / dataset / task / "input_data.jsonl"
        with data_path.open("r", encoding="utf-8") as fh:
            raw_rows = [json.loads(line) for line in fh if line.strip()]

        task_obj = tasks_by_name[task_name]
        assert (
            task_obj._spec["extractor"].__name__ == upstream["extraction_func"].lower()
        )
        eval_docs = list(task_obj.test_docs())
        assert len(eval_docs) == len(raw_rows)
        assert (
            task_obj.config.generation_kwargs["max_gen_toks"]
            == upstream["max_new_tokens"]
        )

        # Verify independent SHA-256 golden digest for representative tasks.
        if task_name in PINNED_COMMIT_GOLDEN_HASHES:
            expected_uuid, expected_sha = PINNED_COMMIT_GOLDEN_HASHES[task_name]
            assert eval_docs[0]["uuid"] == expected_uuid
            actual_sha = hashlib.sha256(
                task_obj.doc_to_text(eval_docs[0]).encode("utf-8")
            ).hexdigest()
            assert actual_sha == expected_sha

        for idx in [0, len(raw_rows) // 2, len(raw_rows) - 1]:
            raw_item = raw_rows[idx]
            expected_prompt = raw_template
            for placeholder, data_key in upstream["prompt_fields"].items():
                if data_key in raw_item:
                    expected_prompt = expected_prompt.replace(
                        placeholder, raw_item[data_key].strip()
                    )
            assert task_obj.doc_to_text(eval_docs[idx]) == expected_prompt

            raw_ans = raw_item[upstream["answer_key"]]
            expected_target = (
                ",".join(str(x) for x in raw_ans)
                if isinstance(raw_ans, list)
                else str(raw_ans)
            )
            assert task_obj.doc_to_target(eval_docs[idx]) == expected_target
