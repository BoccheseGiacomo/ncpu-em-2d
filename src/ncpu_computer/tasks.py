from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import torch

from .tape import encode_strings, integer_to_binary, validate_symbols


@dataclass(frozen=True)
class StringExample:
    input: str
    target: str


@dataclass(frozen=True)
class StringTask:
    name: str
    examples: tuple[StringExample, ...]
    output_mode: str = "single"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task name cannot be empty")
        if not self.examples:
            raise ValueError("a task must contain at least one example")
        if self.output_mode not in {"single", "multiple"}:
            raise ValueError("output_mode must be 'single' or 'multiple'")
        for example in self.examples:
            validate_symbols(example.input, allow_empty=True)
            validate_symbols(example.target, allow_empty=True)
            if example.target and (
                example.target.startswith("B") or example.target.endswith("B")
            ):
                raise ValueError("targets cannot begin or end with B")
            if self.output_mode == "single" and "B" in example.target:
                raise ValueError("single-output targets cannot contain B")
            if self.output_mode == "multiple" and example.target:
                if "BB" in example.target:
                    raise ValueError("multiple-output values use single-B separators")
                if any(not part for part in example.target.split("B")):
                    raise ValueError("multiple-output targets contain an empty value")


def addition_task(operand_bits: int) -> StringTask:
    if operand_bits < 1:
        raise ValueError("operand_bits must be positive")
    limit = 1 << operand_bits
    examples = tuple(
        StringExample(
            input=f"{integer_to_binary(a)}B{integer_to_binary(b)}",
            target=integer_to_binary(a + b),
        )
        for a in range(limit)
        for b in range(limit)
    )
    return StringTask(name=f"addition-{operand_bits}-bit", examples=examples)


def binary_strings(
    max_length: int,
    *,
    include_shorter: bool = True,
    include_empty: bool = True,
) -> tuple[str, ...]:
    if type(max_length) is not int or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")
    if max_length == 0:
        return ("",)
    if include_empty and not include_shorter:
        raise ValueError("include_empty requires include_shorter")
    lengths = range(1, max_length + 1) if include_shorter else (max_length,)
    strings = [""] if include_empty else []
    strings.extend(
        "".join(bits) for length in lengths for bits in product("01", repeat=length)
    )
    return tuple(strings)


def reverse_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    strings = binary_strings(
        max_length,
        include_shorter=include_shorter,
        include_empty=include_shorter,
    )
    return StringTask(
        name="reverse",
        examples=tuple(StringExample(value, value[::-1]) for value in strings),
    )


def bitwise_not_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    strings = binary_strings(
        max_length,
        include_shorter=include_shorter,
        include_empty=include_shorter,
    )
    return StringTask(
        name="bit_not",
        examples=tuple(
            StringExample(
                value,
                "".join("1" if bit == "0" else "0" for bit in value),
            )
            for value in strings
        ),
    )


def parity_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    strings = binary_strings(
        max_length,
        include_shorter=include_shorter,
        include_empty=include_shorter,
    )
    return StringTask(
        name="parity",
        examples=tuple(
            StringExample(value, "1" if value.count("1") % 2 else "0")
            for value in strings
        ),
    )


def copy_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    strings = binary_strings(
        max_length,
        include_shorter=include_shorter,
        include_empty=include_shorter,
    )
    return StringTask(
        name="copy",
        examples=tuple(StringExample(value, value) for value in strings),
    )


def _transformed_task(
    name: str,
    max_length: int,
    transform: Callable[[str], str],
    *,
    include_shorter: bool,
) -> StringTask:
    strings = binary_strings(
        max_length,
        include_shorter=include_shorter,
        include_empty=include_shorter,
    )
    return StringTask(
        name=name,
        examples=tuple(StringExample(value, transform(value)) for value in strings),
    )


def reverse_not_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _transformed_task(
        "reverse_not",
        max_length,
        lambda value: "".join("1" if bit == "0" else "0" for bit in value[::-1]),
        include_shorter=include_shorter,
    )


def shift_left_zero_task(
    max_length: int, *, include_shorter: bool = True
) -> StringTask:
    return _transformed_task(
        "shift_left_zero",
        max_length,
        lambda value: value[1:] + "0" if value else "",
        include_shorter=include_shorter,
    )


def shift_right_zero_task(
    max_length: int, *, include_shorter: bool = True
) -> StringTask:
    return _transformed_task(
        "shift_right_zero",
        max_length,
        lambda value: "0" + value[:-1] if value else "",
        include_shorter=include_shorter,
    )


def _gray_encode(value: str) -> str:
    if not value:
        return ""
    return value[0] + "".join(
        "1" if left != right else "0" for left, right in zip(value, value[1:])
    )


def gray_encode_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _transformed_task(
        "gray_encode",
        max_length,
        _gray_encode,
        include_shorter=include_shorter,
    )


def _prefix_xor(value: str) -> str:
    parity = 0
    output = []
    for bit in value:
        parity ^= int(bit)
        output.append(str(parity))
    return "".join(output)


def prefix_xor_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _transformed_task(
        "prefix_xor",
        max_length,
        _prefix_xor,
        include_shorter=include_shorter,
    )


def _increment(value: str) -> str:
    bits = list(value)
    for index in range(len(bits) - 1, -1, -1):
        if bits[index] == "0":
            bits[index] = "1"
            break
        bits[index] = "0"
    return "".join(bits)


def increment_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return _transformed_task(
        "increment",
        max_length,
        _increment,
        include_shorter=include_shorter,
    )


def append_task(
    max_length: int, bit: str, *, include_shorter: bool = True
) -> StringTask:
    if bit not in {"0", "1"}:
        raise ValueError("appended bit must be '0' or '1'")
    strings = binary_strings(
        max_length,
        include_shorter=include_shorter,
        include_empty=include_shorter,
    )
    return StringTask(
        name=f"append_{bit}",
        examples=tuple(StringExample(value, value + bit) for value in strings),
    )


def append_zero_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return append_task(max_length, "0", include_shorter=include_shorter)


def append_one_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    return append_task(max_length, "1", include_shorter=include_shorter)


TASK_FACTORIES = {
    "copy": copy_task,
    "bit_not": bitwise_not_task,
    "reverse": reverse_task,
    "reverse_not": reverse_not_task,
    "shift_left_zero": shift_left_zero_task,
    "shift_right_zero": shift_right_zero_task,
    "gray_encode": gray_encode_task,
    "prefix_xor": prefix_xor_task,
    "increment": increment_task,
    "parity": parity_task,
    "append_0": append_zero_task,
    "append_1": append_one_task,
}


def binary_tasks(
    names: tuple[str, ...], max_length: int, *, include_shorter: bool = True
) -> tuple[StringTask, ...]:
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("task names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("task names must be non-empty and unique")
    unknown = set(names) - set(TASK_FACTORIES)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    return tuple(
        TASK_FACTORIES[name](max_length, include_shorter=include_shorter)
        for name in names
    )


@dataclass(frozen=True)
class TaskDataset:
    name: str
    output_mode: str
    input_strings: tuple[str, ...]
    target_strings: tuple[str, ...]
    inputs: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor

    @classmethod
    def from_task(cls, task: StringTask, tape_slots: int) -> "TaskDataset":
        inputs = tuple(example.input for example in task.examples)
        targets = tuple(example.target for example in task.examples)
        maximum_length = tape_slots - 1
        if max(map(len, inputs)) > maximum_length:
            raise ValueError(
                f"task {task.name!r} has an input that does not leave one blank "
                "tape cell"
            )
        if max(map(len, targets)) > maximum_length:
            raise ValueError(
                f"task {task.name!r} has an output that does not leave one blank "
                "tape cell"
            )
        return cls(
            name=task.name,
            output_mode=task.output_mode,
            input_strings=inputs,
            target_strings=targets,
            inputs=encode_strings(inputs, tape_slots),
            targets=encode_strings(targets, tape_slots),
            target_lengths=torch.tensor(
                [len(target) for target in targets], dtype=torch.int64
            ),
        )

    def __post_init__(self) -> None:
        count = len(self.input_strings)
        if count < 1 or len(self.target_strings) != count:
            raise ValueError("dataset strings have inconsistent lengths")
        if self.inputs.ndim != 2 or self.targets.shape != self.inputs.shape:
            raise ValueError("inputs and targets must be matching rank-two tensors")
        if self.inputs.shape[0] != count:
            raise ValueError("tensor and string example counts differ")
        if not torch.is_floating_point(self.inputs) or not torch.is_floating_point(
            self.targets
        ):
            raise ValueError("inputs and targets must be floating point")
        if self.target_lengths.shape != (count,):
            raise ValueError("target_lengths has the wrong shape")

    def __len__(self) -> int:
        return len(self.input_strings)

    @property
    def tape_slots(self) -> int:
        return self.inputs.shape[1]

    @property
    def signature(self) -> str:
        data = {
            "name": self.name,
            "output_mode": self.output_mode,
            "inputs": self.input_strings,
            "targets": self.target_strings,
        }
        encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def take(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        indices = torch.as_tensor(indices, dtype=torch.int64, device="cpu")
        return (
            self.inputs[indices],
            self.targets[indices],
            self.target_lengths[indices],
        )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = torch.randint(len(self), (batch_size,), generator=generator)
        return self.take(indices)


@dataclass(frozen=True)
class MultiTaskDataset:
    datasets: tuple[TaskDataset, ...]

    @classmethod
    def from_tasks(
        cls, tasks: tuple[StringTask, ...], tape_slots: int
    ) -> "MultiTaskDataset":
        return cls(tuple(TaskDataset.from_task(task, tape_slots) for task in tasks))

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("at least one task dataset is required")
        if len(set(self.task_names)) != len(self.datasets):
            raise ValueError("task dataset names must be unique")
        first = self.datasets[0]
        for dataset in self.datasets[1:]:
            if dataset.tape_slots != first.tape_slots:
                raise ValueError("all tasks must use the same tape capacity")
            if dataset.input_strings != first.input_strings:
                raise ValueError("all tasks must use the same ordered inputs")

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(dataset.name for dataset in self.datasets)

    @property
    def tape_slots(self) -> int:
        return self.datasets[0].tape_slots

    @property
    def signatures(self) -> tuple[str, ...]:
        return tuple(dataset.signature for dataset in self.datasets)

    def __len__(self) -> int:
        return sum(map(len, self.datasets))

    def __getitem__(self, task_name: str) -> TaskDataset:
        try:
            index = self.task_names.index(task_name)
        except ValueError as error:
            raise KeyError(task_name) from error
        return self.datasets[index]

    def balanced_sample(
        self, batch_size_per_task: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        if type(batch_size_per_task) is not int or batch_size_per_task < 1:
            raise ValueError("batch_size_per_task must be a positive integer")
        batches = []
        for task_index, dataset in enumerate(self.datasets):
            inputs, targets, lengths = dataset.sample(batch_size_per_task, generator)
            task_indices = torch.full(
                (batch_size_per_task,), task_index, dtype=torch.int64
            )
            batches.append((inputs, targets, lengths, task_indices))
        combined = tuple(torch.cat(parts) for parts in zip(*batches))
        permutation = torch.randperm(combined[0].shape[0], generator=generator)
        return tuple(value[permutation] for value in combined)


def semantic_correct(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_lengths: torch.Tensor,
    output_mode: str,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device)
    target_lengths = torch.as_tensor(
        target_lengths, dtype=torch.int64, device=prediction.device
    )
    squeeze_time = prediction.ndim == 2
    if squeeze_time:
        prediction = prediction.unsqueeze(1)
    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape (batch, time, tape) or (batch, tape)"
        )
    batch, _, tape_slots = prediction.shape
    if target.shape != (batch, tape_slots):
        raise ValueError("target shape does not match prediction")
    if target_lengths.shape != (batch,):
        raise ValueError("target_lengths shape does not match prediction")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    terminator_width = 1 if output_mode == "single" else 2
    required_end = torch.minimum(
        target_lengths + terminator_width,
        torch.full_like(target_lengths, tape_slots),
    )
    positions = torch.arange(tape_slots, device=prediction.device).view(1, 1, -1)
    required = positions < required_end.view(batch, 1, 1)
    correct = ((prediction == target.unsqueeze(1)) | ~required).all(dim=-1)
    return correct.squeeze(1) if squeeze_time else correct
