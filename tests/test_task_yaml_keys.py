from pathlib import Path

import yaml


TASKS_DIR = Path(__file__).parent.parent / "lm_eval" / "tasks"


class _Loader(yaml.SafeLoader):
    pass


# task configs use custom tags such as !function; the values don't matter here
_Loader.add_multi_constructor("!", lambda loader, suffix, node: None)


def _task_configs():
    for path in sorted(TASKS_DIR.rglob("*")):
        if path.is_file() and (
            path.suffix in (".yaml", ".yml") or path.name.endswith("_yaml")
        ):
            yield path


def _duplicate_keys(node):
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if key.value in seen:
                yield key.value, key.start_mark.line + 1
            seen.add(key.value)
            yield from _duplicate_keys(value)
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            yield from _duplicate_keys(value)


def test_task_configs_have_no_duplicate_keys():
    # PyYAML keeps the last value for a repeated key, so the first one is silently ignored
    found = []
    for path in _task_configs():
        try:
            node = yaml.compose(path.read_text(encoding="utf-8"), Loader=_Loader)
        except yaml.YAMLError:
            continue
        for key, line in _duplicate_keys(node):
            found.append(f"{path.relative_to(TASKS_DIR)}:{line} {key}")
    assert not found, "duplicate keys:\n" + "\n".join(found)
