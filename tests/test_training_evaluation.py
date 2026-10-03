from dataclasses import replace

import pytest
import torch

from ncpu_computer import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TapeLayout,
    TestCase,
    Trainer,
    TrainingConfig,
    evaluate_cases,
    infer,
    learning_rate_at,
    load_model,
    perception_noise,
    sample_trials,
    supervised_loss,
    validate_task_specs,
)


TASK_SPECS = (("copy", 1.0), ("bit_not", 1.0))


def tiny_config(
    *, updates=2, mode="zero", train_rule=True, train_program=False, fire_rate=1.0
):
    return ExperimentConfig(
        geometry=GeometryConfig(stride=2, vertical_space=1, horizontal_space=2),
        model=ModelConfig(
            hidden_size=4,
            fixed_kernels=("identity",),
            learnable_kernels=0,
            program_mode=mode,
            fire_rate=fire_rate,
            max_abs_state=None,
        ),
        training=TrainingConfig(
            updates=updates,
            batch_size_per_task=2,
            base_tape_slots=(3, 4),
            base_input_max_lengths=(1, 2),
            tape_variation=0.0,
            input_variation=0.0,
            free_steps_per_tape_slot=0.4,
            time_variation=0.0,
            supervision_ratio=1.0,
            train_rule=train_rule,
            train_program=train_program,
            validation_every=1,
            checkpoint_every=1,
            device="cpu",
        ),
        test_cases=(TestCase("tiny", 3, 1, 1, 1),),
    )


def test_paired_trials_are_tape_timed_and_reproducible():
    config = tiny_config().training
    first = sample_trials(config, ("copy",), torch.Generator().manual_seed(9))
    second = sample_trials(config, ("copy",), torch.Generator().manual_seed(9))
    assert first == second
    assert {(trial.tape_slots, trial.input_max_length) for trial in first} == {
        (3, 1),
        (4, 2),
    }
    assert all(trial.input_max_length <= trial.tape_slots - 1 for trial in first)
    assert sorted(trial.free_steps for trial in first) == [1, 2]


def test_invalid_input_resamples_without_resampling_tape(monkeypatch):
    import ncpu_computer.training as training_module

    config = replace(
        tiny_config().training,
        base_tape_slots=(4,),
        base_input_max_lengths=(2,),
        tape_variation=0.25,
        input_variation=0.5,
        free_steps_per_tape_slot=2.0,
    )
    values = iter((3, 3, 1, 6))
    bases = []

    def fake_sample(base, variation, generator):
        bases.append(base)
        return next(values)

    monkeypatch.setattr(training_module, "_sample_varied", fake_sample)
    trial = sample_trials(config, ("append_0",), torch.Generator().manual_seed(0))[0]
    assert bases == [4, 2, 2, 6.0]
    assert trial == training_module.TrialSpec(3, 1, 6, 6)


def test_supervised_loss_uses_every_tape_cell_and_weighted_task_means():
    layout = TapeLayout(GeometryConfig(), 2)
    rollout = torch.zeros(3, 2, 3, layout.height, layout.width)
    rollout[0, 1, 1, layout.tape_row, layout.tape_slice] = 1.0
    rollout[1:, 1, 1, layout.tape_row, layout.tape_slice] = 3.0
    rollout[:, 1, 1, 0, 0] = 1000.0
    losses = supervised_loss(
        rollout,
        torch.zeros(3, 2),
        layout,
        1,
        0,
        1,
        torch.tensor([0, 1, 1]),
        2,
        (1.0, 3.0),
    )
    assert losses.per_task.tolist() == pytest.approx([1.0, 9.0])
    assert float(losses.total) == pytest.approx(7.0)


def test_task_specs_require_unique_names_and_positive_finite_weights():
    assert validate_task_specs((("copy", 1), ("reverse", 2.0))) == (
        ("copy", "reverse"),
        (1.0, 2.0),
    )
    for specs in (
        (("copy", 0.0),),
        (("copy", float("inf")),),
        (("copy", 1.0), ("copy", 2.0)),
    ):
        with pytest.raises(ValueError):
            validate_task_specs(specs)


def test_training_accumulates_trials_and_optimizer_respects_modes():
    rule_only = Trainer(tiny_config(), TASK_SPECS)
    metrics = rule_only.train_step()
    assert len(metrics.trials) == 2
    assert [group["name"] for group in rule_only.optimizer.param_groups] == ["rule"]
    assert all(
        not parameter.requires_grad for parameter in rule_only.model.program_parameters
    )

    program_config = tiny_config(
        mode="learned_read_only", train_rule=False, train_program=True
    )
    program_only = Trainer(program_config, TASK_SPECS)
    with torch.no_grad():
        program_only.model.rule.hidden.weight.fill_(0.1)
        program_only.model.rule.hidden.bias.fill_(0.1)
        program_only.model.rule.output.weight.fill_(0.05)
    before = program_only.model.programs.detach().clone()
    program_only.train_step()
    assert [group["name"] for group in program_only.optimizer.param_groups] == [
        "program"
    ]
    assert not torch.equal(before, program_only.model.programs)

    joint = Trainer(
        tiny_config(mode="learned_mutable", train_rule=True, train_program=True),
        TASK_SPECS,
    )
    assert [group["name"] for group in joint.optimizer.param_groups] == [
        "rule",
        "program",
    ]
    assert joint.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        joint.config.training.weight_decay
    )
    assert joint.optimizer.param_groups[1]["weight_decay"] == pytest.approx(
        joint.config.training.program_weight_decay
    )


def test_noise_anneals_linearly():
    training = replace(
        tiny_config(updates=3).training,
        perception_noise_start=0.1,
        perception_noise_end=0.0,
    )
    assert [perception_noise(training, i) for i in range(3)] == pytest.approx(
        [0.1, 0.05, 0.0]
    )


def test_staged_cosine_schedule_hits_every_anchor_and_plateau():
    training = replace(
        tiny_config(updates=101).training,
        lr_points=TrainingConfig().lr_points,
    )
    expected = {
        0: 2e-3,
        35: 2e-3,
        60: 7e-4,
        75: 7e-4,
        95: 1e-4,
        100: 1e-4,
    }
    assert {
        update: learning_rate_at(training, update) for update in expected
    } == pytest.approx(expected)


def test_linear_schedule_supports_growth_and_decay():
    training = replace(
        tiny_config(updates=5).training,
        lr_points=((0.0, 1.0), (0.5, 3.0), (1.0, 1.0)),
        lr_interpolation="linear",
    )
    assert [learning_rate_at(training, update) for update in range(5)] == pytest.approx(
        [1.0, 2.0, 3.0, 2.0, 1.0]
    )


def test_cosine_schedule_smoothly_interpolates_between_anchors():
    training = replace(
        tiny_config(updates=5).training,
        lr_points=((0.0, 1.0), (1.0, 3.0)),
        lr_interpolation="cosine",
    )
    assert [learning_rate_at(training, update) for update in range(5)] == pytest.approx(
        [1.0, 1.0 + (1.0 - 2**-0.5), 2.0, 2.0 + 2**-0.5, 3.0]
    )


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config = replace(
        tiny_config(fire_rate=0.5),
        training=replace(
            tiny_config(fire_rate=0.5).training,
            perception_noise_start=0.02,
            perception_noise_end=0.01,
        ),
    )
    uninterrupted = Trainer(config, TASK_SPECS)
    uninterrupted.train_step()
    expected_second = uninterrupted.train_step()
    interrupted = Trainer(config, TASK_SPECS)
    interrupted.train_step()
    path = tmp_path / "resume.pt"
    interrupted.save(path)
    with pytest.raises(ValueError, match="task specs do not match"):
        Trainer.from_checkpoint(path, (("copy", 1.0), ("bit_not", 2.0)), device="cpu")
    resumed = Trainer.from_checkpoint(path, TASK_SPECS, device="cpu")
    actual_second = resumed.train_step()
    assert actual_second.trials == expected_second.trials
    assert actual_second.loss == pytest.approx(expected_second.loss)
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_format_seven_checkpoint_migrates_without_changing_dynamics(tmp_path):
    trainer = Trainer(tiny_config(), TASK_SPECS)
    checkpoint = trainer.checkpoint()
    checkpoint["format_version"] = 7
    del checkpoint["task_specs"]
    del checkpoint["config"]["model"]["activation"]
    del checkpoint["config"]["model"]["state_leak"]
    path = tmp_path / "format-seven.pt"
    torch.save(checkpoint, path)

    resumed = Trainer.from_checkpoint(path, TASK_SPECS, device="cpu")
    assert resumed.task_specs == TASK_SPECS
    assert resumed.config.model.activation == "relu"
    assert resumed.config.model.state_leak == 0.0
    model, loaded_config, loaded_checkpoint = load_model(path, "cpu")
    assert model.config == resumed.config.model
    assert loaded_config == resumed.config
    assert loaded_checkpoint["task_specs"] == TASK_SPECS


def test_checkpoint_and_variable_case_evaluation(tmp_path):
    config = tiny_config(updates=1)
    trainer = Trainer(config, TASK_SPECS)
    trainer.fit(tmp_path, progress_every=1)
    model, loaded, checkpoint = load_model(tmp_path / "best.pt", "cpu")
    assert loaded == config
    assert checkpoint["format_version"] == 8
    assert checkpoint["task_specs"] == TASK_SPECS
    assert len(checkpoint["task_signatures"]) == 2
    assert set(trainer.history[-1]["validation_accuracies"]) == {
        "copy",
        "bit_not",
    }
    results = evaluate_cases(
        model,
        config.geometry,
        ("copy", "bit_not"),
        config.test_cases,
        batch_size=2,
    )
    assert [result.task for result in results] == ["copy", "bit_not", "aggregate"]
    inferred = infer(
        model, config.geometry, "10", tape_slots=5, task_name="copy", steps=0
    )
    assert inferred.values.tolist() == [1.0, -1.0, 0.0, 0.0, 0.0]
