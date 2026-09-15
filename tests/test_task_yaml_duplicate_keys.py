"""A task YAML must not declare the same key twice.

PyYAML keeps the last value and reports nothing, so a repeated key drops
configuration silently: `turblimp_group.yaml` and `zhoblimp_group.yaml` lost
their entire `acc` aggregate metric that way. An overridden `!function` is
worse than silent, because the tag is still resolved while the document is
parsed, so a stale reference raises even though the line it sits on is dead.
"""

import collections
from pathlib import Path

import yaml


TASKS_DIR = Path(__file__).parent.parent / "lm_eval" / "tasks"


def _duplicate_keys(node: yaml.Node) -> set[str]:
    """Keys declared more than once in the same mapping, at any depth."""
    duplicates: set[str] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, yaml.MappingNode):
            counts = collections.Counter(key.value for key, _ in current.value)
            duplicates.update(key for key, count in counts.items() if count > 1)
            pending.extend(value for _, value in current.value)
        elif isinstance(current, yaml.SequenceNode):
            pending.extend(current.value)
    return duplicates


def test_no_task_yaml_declares_a_key_twice():
    offenders = {}
    for path in sorted(TASKS_DIR.rglob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            try:
                # compose() builds nodes without constructing objects, so
                # `!function` targets are never imported and a task's optional
                # dependencies do not have to be installed to run this.
                document = yaml.compose(handle, Loader=yaml.SafeLoader)
            except yaml.YAMLError:
                continue  # malformed yaml is not this test's concern
        if document is None:
            continue
        duplicates = _duplicate_keys(document)
        if duplicates:
            offenders[str(path.relative_to(TASKS_DIR))] = sorted(duplicates)

    assert not offenders, "duplicate keys silently drop config:\n" + "\n".join(
        f"  {name}: {', '.join(keys)}" for name, keys in sorted(offenders.items())
    )
