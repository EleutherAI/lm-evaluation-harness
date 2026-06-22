# From https://github.com/princeton-nlp/collie
#
# Upstream License:
#   MIT License
#   Copyright (c) 2023 Howard Chen
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from nltk import sent_tokenize, word_tokenize


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any, Literal, Protocol

    OperandStr = Literal["==", "!=", "<", ">", "<=", ">=", "in", "not in"]
    ReductionStr = Literal["at least", "at most", "exactly", "all", "any"]
    LevelStr = Literal[
        "character", "word", "phrase", "sentence", "paragraph", "passage"
    ]

    class _Checkable(Protocol):
        """Structural type for constraint-like callables (Constraint / Logic).

        Sub-constraints stored in `And`/`Or`/`All` are invoked with either one
        argument (`x`) or two (`x, target`) and expose `extract`.
        """

        def __call__(self, x: Any, target: Any = ...) -> Any: ...

        def check(self, text: Any, target: Any) -> bool: ...

        def extract(self, x: Any) -> Any: ...


def sus_target(t: str) -> bool:
    """Returns True if the target t is suspicious."""
    if not isinstance(t, str):
        return False
    t = t.strip()  # leading / trailing whitespace ok
    if t[0] not in string.ascii_letters + string.digits:
        return True  # first letter not a letter
    return t[-1] not in string.ascii_letters + string.digits + string.punctuation


class Level:
    """Level Base class."""

    _para_delim = "\n\n"

    def __init__(self, level: LevelStr | None = None) -> None:
        self.level = level

    def __call__(self, text: Any) -> Any:
        if self.level is None:
            return text
        if isinstance(text, str):
            tokenized = None
            if self.level == "character":
                tokenized = list(text)
            elif self.level == "word":
                tokenized = [
                    x for x in word_tokenize(text) if x not in string.punctuation
                ]
            elif self.level == "phrase":
                raise NotImplementedError
            elif self.level == "sentence":
                tokenized = sent_tokenize(text)
            elif self.level == "paragraph":
                tokenized = self.split_paragraphs(text)
            elif self.level == "passage":
                raise NotImplementedError
            # TODO: make this more general
            tokenized = (
                [tok.strip().strip(".") for tok in tokenized]
                if tokenized is not None
                else None
            )
            return tokenized
        elif isinstance(text, list):
            return [self(unit) for unit in text]
        else:
            raise ValueError(
                f"Input text must be a string or a list of strings, not {type(text)}."
            )

    @staticmethod
    def join_paragraphs(text: Iterable[str]) -> str:
        return Level._para_delim.join(text)

    @staticmethod
    def split_paragraphs(text: str) -> list[str]:
        return text.split(Level._para_delim)


class InputLevel(Level):
    def __str__(self) -> str:
        return f"InputLevel({self.level})"


class TargetLevel(Level):
    def __str__(self) -> str:
        return f"TargetLevel({self.level})"


class Transformation:
    """Base class for transformations"""

    pass


class Count(Transformation):
    def __init__(self, count_target: str | None = None) -> None:
        super().__init__()
        self.count_target = count_target

    def __call__(self, units: list) -> int:
        if self.count_target is None:
            count = len(units)
        else:
            count = len([unit for unit in units if unit == self.count_target])
        return count

    def __str__(self) -> str:
        if self.count_target is None:
            return "Count()"
        else:
            return f"Count({self.count_target})"


class Position(Transformation):
    def __init__(self, position: int | list[int] | None = None) -> None:
        super().__init__()
        self.position = position

    @staticmethod
    def get(units: list, position: int) -> Any:
        if len(units) == 0:
            return None
        if position >= len(units):
            return None
        if position < -len(units):
            return None
        return units[position]

    def __call__(self, units: list) -> Any:
        if isinstance(self.position, int):
            return self.get(units, self.position)
        elif isinstance(self.position, list):
            return [self.get(units, i) for i in self.position]
        else:
            raise ValueError(
                f"Position must be an integer or a list of integers, not {type(self.position)}."
            )

    def __str__(self) -> str:
        return f"Position({self.position})"


class PositionOf(Transformation):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def __call__(self, x: Any) -> list[int]:
        units = self.func(x)
        return [i for i, u in enumerate(units)]

    def __str__(self) -> str:
        return f"PositionsOf({self.func})"


class MapPosition(Transformation):
    def __init__(self, func: Callable, map_pos_func: Callable) -> None:
        super().__init__()
        self.func = func
        self.map_pos_func = map_pos_func

    def __call__(self, x: Any) -> Any:
        units = self.func(x)
        return self.map_pos_func(units)

    def __str__(self) -> str:
        return f"MapPosition({self.func}, {self.map_pos_func})"


class Aggregate(Transformation):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def __call__(self, x: Any) -> str:
        # TODO: need to merge with the level class
        units = self.func(x)
        return "".join(units)

    def __str__(self) -> str:
        return f"Aggregate({self.func})"


class Max(Transformation):
    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def __call__(self, x: Any) -> Any:
        units = self.func(x)
        return max(units)

    def __str__(self) -> str:
        return f"Max({self.func})"


class ForEach(Transformation):
    def __init__(self, func: Any) -> None:
        self.func = func

    def __call__(self, units: list) -> list:
        if self.func is Ellipsis:
            func = lambda x: x
        else:
            func = self.func
        return [func(unit) for unit in units]

    def __str__(self) -> str:
        if self.func is Ellipsis:
            return "ForEach(...)"
        else:
            return f"ForEach({self.func})"


class Logic:
    """Base class for composite constraints; subclasses implement `__call__`."""

    def __call__(self, x: Any, target: Any = None) -> bool:
        raise NotImplementedError("Logic subclasses must implement __call__")

    def check(self, x: Any, target: Any) -> bool:
        return self(x, target)

    def extract(self, x: Any) -> Any:
        raise NotImplementedError("Logic subclasses must implement extract")


class And(Logic):
    def __init__(self, callable_1: _Checkable, callable_2: _Checkable) -> None:
        super().__init__()
        self.callable_1 = callable_1
        self.callable_2 = callable_2
        self.callables = [callable_1, callable_2]

    def __call__(self, x: Any, target: Any = None) -> bool:
        if target is None:
            return self.callable_1(x) and self.callable_2(x)
        elif isinstance(target, list):
            assert len(target) == 2
            return self.callable_1(x, target[0]) and self.callable_2(x, target[1])
        else:
            return self.callable_1(x, target) and self.callable_2(x, target)

    def __str__(self) -> str:
        return f"And({self.callable_1}, {self.callable_2})"

    def extract(self, x: Any) -> list:
        return [callable_.extract(x) for callable_ in self.callables]


class Or(Logic):
    def __init__(self, callable_1: _Checkable, callable_2: _Checkable) -> None:
        super().__init__()
        self.callable_1 = callable_1
        self.callable_2 = callable_2
        self.callables = [callable_1, callable_2]

    def __call__(self, x: Any, target: Any = None) -> bool:
        if target is None:
            return self.callable_1(x) or self.callable_2(x)
        elif isinstance(target, list):
            assert len(target) == 2
            return self.callable_1(x, target[0]) or self.callable_2(x, target[1])
        else:
            return self.callable_1(x, target) or self.callable_2(x, target)

    def __str__(self) -> str:
        return f"Or({self.callable_1}, {self.callable_2})"

    def extract(self, x: Any) -> list:
        return [callable_.extract(x) for callable_ in self.callables]


class All(Logic):
    def __init__(self, *callables: _Checkable) -> None:
        super().__init__()
        self.callables = callables

    def __call__(self, x: Any, target: Any = None) -> bool:
        if target is None:
            return all(callable_(x) for callable_ in self.callables)
        elif isinstance(target, list):
            return all(
                callable_(x, t)
                for callable_, t in zip(self.callables, target, strict=False)
            )
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return f"All({', '.join([str(c) for c in self.callables])})"

    def extract(self, x: Any) -> list:
        return [callable_.extract(x) for callable_ in self.callables]


class Relation:
    """
    Abstract relation class that works for more literal types.
    """

    def __init__(self, operand: OperandStr) -> None:
        self.operand = operand

    def _patch_literal(self, literal: Any) -> Any:
        # apply transformations on literal to remove artifacts and casing

        def _patch(literal: Any) -> Any:
            # apply the patch if it is
            if not isinstance(literal, str):
                return literal
            stripped = literal.lower().strip(string.punctuation + " ")
            return literal if stripped == "" else stripped

        if isinstance(literal, list):
            return [_patch(x) for x in literal]

        return _patch(literal)

    def __call__(self, literal_1: Any, literal_2: Any) -> bool:
        literal_1, literal_2 = (
            self._patch_literal(literal_1),
            self._patch_literal(literal_2),
        )
        if self.operand in ["==", "!=", "<", ">", "<=", ">="]:
            if isinstance(literal_2, list) and len(literal_2) == 1:
                literal_2 = literal_2[0]
            if self.operand == "==":
                return literal_1 == literal_2
            elif self.operand == "!=":
                return literal_1 != literal_2
            elif self.operand == "<":
                return literal_1 < literal_2
            elif self.operand == "<=":
                return literal_1 <= literal_2
            elif self.operand == ">":
                return literal_1 > literal_2
            else:  # ">="
                return literal_1 >= literal_2
        elif self.operand in ["in", "not in"]:
            if self.operand == "in":
                if isinstance(literal_2, list):
                    return all(l in literal_1 for l in literal_2)
                else:
                    return literal_2 in literal_1
            else:  # "not in"
                if isinstance(literal_2, list):
                    return not any(l in literal_1 for l in literal_2)
                else:
                    return literal_2 not in literal_1
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return f"Relation({self.operand})"


class Reduction:
    def __init__(
        self, reduction: ReductionStr | None = None, value: int | None = None
    ) -> None:
        """
        Reduction (str): one of ['at least', 'at most', 'exactly', 'all', 'any']
        """
        self.reduction = reduction
        self.value = value

    def __call__(self, x: Any, target: Any, relation: Relation) -> bool:
        if self.reduction is None:
            return relation(x, target)
        if not isinstance(target, list):
            target = [target] * len(x)
        if len(x) != len(target):
            return False
        # assert len(x) == len(target), f'Length of x ({len(x)}) and target ({len(target)}) must be the same.'
        results = [
            relation(x_i, target_i) for x_i, target_i in zip(x, target, strict=False)
        ]

        if self.reduction == "all":
            return all(results)
        elif self.reduction == "any":
            return any(results)
        elif self.reduction == "at least":
            assert self.value is not None
            return sum(results) >= self.value
        elif self.reduction == "at most":
            assert self.value is not None
            return sum(results) <= self.value
        elif self.reduction == "exactly":
            return sum(results) == self.value
        else:
            raise ValueError(f"Unknown reduction: {self.reduction!r}")

    def __str__(self) -> str:
        if self.value is not None:
            return f"Reduction({self.reduction} {self.value})"
        else:
            return f"Reduction({self.reduction})"


class Constraint:
    def __init__(
        self,
        input_level: InputLevel | None = None,
        target_level: TargetLevel | None = None,
        transformation: Callable | None = None,
        relation: Relation | None = None,
        reduction: Reduction | None = None,
    ) -> None:
        self.input_level = input_level or InputLevel()
        self.target_level = target_level or TargetLevel()
        self.transformation = transformation
        self.relation = relation
        self.reduction = reduction or Reduction()

    def extract(self, text: Any) -> Any:
        if self.input_level is not None:
            input_units = self.input_level(text)
        else:
            input_units = text
        x = self.target_level(input_units)
        if self.transformation is not None:
            x = self.transformation(x)
        return x

    def check(self, text: Any, target: Any) -> bool:
        x = self.extract(text)
        assert self.relation is not None
        return self.reduction(x, target, self.relation)

    def __call__(self, text: Any, target: Any) -> bool:
        return self.check(text, target)

    def __str__(self) -> str:
        return (
            f"Constraint(\n"
            f"    {self.input_level},\n"
            f"    {self.target_level},\n"
            f"    Transformation({self.transformation}),\n"
            f"    {self.relation},\n"
            f"    {self.reduction}\n"
            f")"
        )

    def __repr__(self) -> str:
        return self.__str__()
