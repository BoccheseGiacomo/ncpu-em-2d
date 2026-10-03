import pytest
import torch
from torch.nn import functional as F

from ncpu_computer import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    NeuralCellularAutomaton,
    TapeLayout,
    validate_experiment,
)
from ncpu_computer.model import Perception, UpdateRule, perception_kernel


TASKS = ("copy", "reverse")
GEOMETRY = GeometryConfig()


def make_model(config=ModelConfig()):
    torch.manual_seed(0)
    return NeuralCellularAutomaton(config, GEOMETRY, TASKS)


def test_default_model_has_shared_io_and_zero_periodic_program():
    model = make_model()
    assert model.config.channels == 5
    assert model.config.io_channel == 1
    assert model.perception.kernel_count == 4
    assert model.programs.shape == (2, 1, 3, 2)
    assert model.program_parameters == ()
    assert torch.count_nonzero(model.programs) == 0


@pytest.mark.parametrize("program_start", [0, 1])
def test_program_repeats_from_configured_origin(program_start):
    geometry = GeometryConfig(program_start=program_start)
    config = ModelConfig(program_channels=2, program_mode="learned_read_only")
    model = NeuralCellularAutomaton(config, geometry, TASKS)
    with torch.no_grad():
        model.programs.copy_(
            torch.arange(model.programs.numel()).reshape_as(model.programs)
        )
    for slots in (3, 8):
        layout = TapeLayout(geometry, slots)
        grid = model.program_grid(torch.tensor([1]), layout.width)
        if program_start == 1:
            assert torch.count_nonzero(grid[..., 0]) == 0
            assert (layout.width - 1) % geometry.stride == 0
        for x in range(program_start, layout.width):
            phase = (x - program_start) % geometry.stride
            assert torch.equal(grid[0, :, :, x], model.programs[1, :, :, phase])
        if program_start == 0:
            assert layout.width % geometry.stride == 1
        state = model.initial_state(
            layout.render_tape(torch.ones(1, slots)), torch.tensor([1])
        )
        assert torch.equal(state[0, :2], grid[0])
        assert torch.equal(
            layout.extract_tape(state[:, config.io_channel]), torch.ones(1, slots)
        )


@pytest.mark.parametrize(
    "mode,changes",
    [("zero", False), ("learned_read_only", False), ("learned_mutable", True)],
)
def test_program_mutability_modes(mode, changes):
    model = make_model(ModelConfig(program_mode=mode, max_abs_state=None))
    with torch.no_grad():
        model.rule.hidden.weight.zero_()
        model.rule.hidden.bias.fill_(1.0)
        model.rule.output.weight.fill_(0.1)
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(layout.render_tape(torch.ones(1, 3)), torch.tensor([0]))
    updated = model.step(state)
    assert (not torch.equal(updated[:, :1], state[:, :1])) is changes
    if mode == "learned_mutable":
        assert torch.count_nonzero(state[:, :1, :, 0]) == 0
        assert torch.count_nonzero(updated[:, :1, :, 0]) > 0
    assert not torch.equal(
        updated[:, model.config.io_channel], state[:, model.config.io_channel]
    )


def test_zero_delta_initialization_is_identity_at_every_shape():
    model = make_model(ModelConfig(state_leak=0.0))
    for slots in (2, 5):
        layout = TapeLayout(GEOMETRY, slots)
        state = model.initial_state(
            layout.render_tape(torch.randn(2, slots)), torch.tensor([0, 1])
        )
        assert torch.equal(
            model(state, 3), state.unsqueeze(1).expand(-1, 4, -1, -1, -1)
        )


def test_experiment_validation_accepts_masked_leak_dynamics():
    config = ExperimentConfig(
        model=ModelConfig(state_leak=0.2, fire_rate=0.5, max_abs_state=0.75)
    )
    report = validate_experiment(config, ("copy",))
    assert report.task_names == ("copy",)


def test_leak_affects_only_mutable_channels():
    model = make_model(
        ModelConfig(
            program_mode="learned_read_only",
            state_leak=0.25,
            fire_rate=1.0,
            max_abs_state=None,
        )
    )
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(
        layout.render_tape(torch.ones(2, 3)), torch.tensor([0, 1])
    )
    updated = model.step(state)
    program_channels = model.config.program_channels
    assert torch.equal(updated[:, :program_channels], state[:, :program_channels])
    assert torch.equal(
        updated[:, program_channels:], state[:, program_channels:] * 0.75
    )


def test_clipping_does_not_change_frozen_program_channels():
    model = make_model(
        ModelConfig(
            program_mode="learned_read_only",
            state_leak=0.5,
            fire_rate=1.0,
            max_abs_state=1.0,
        )
    )
    layout = TapeLayout(GEOMETRY, 3)
    state = torch.full((1, model.config.channels, layout.height, layout.width), 4.0)
    updated = model.step(state)
    assert torch.equal(updated[:, :1], state[:, :1])
    assert torch.equal(updated[:, 1:], torch.ones_like(updated[:, 1:]))


def test_read_only_program_receives_gradients_through_mutable_dynamics():
    model = make_model(
        ModelConfig(
            program_mode="learned_read_only",
            state_leak=0.1,
            fire_rate=1.0,
            max_abs_state=None,
        )
    )
    with torch.no_grad():
        model.programs.fill_(0.5)
        model.rule.hidden.weight.fill_(0.05)
        model.rule.hidden.bias.fill_(0.1)
        model.rule.output.weight.fill_(0.02)
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(
        layout.render_tape(torch.zeros(1, 3)), torch.tensor([0])
    )
    rollout = model(state, 2)
    assert torch.equal(rollout[:, 0, :1], rollout[:, -1, :1])
    rollout[:, -1, model.config.io_channel].sum().backward()
    assert model.programs.grad is not None
    assert bool(torch.isfinite(model.programs.grad).all())
    assert torch.count_nonzero(model.programs.grad) > 0


def test_fire_mask_applies_to_leak_and_is_fresh_per_rollout():
    model = make_model(ModelConfig(state_leak=0.5, fire_rate=0.5, max_abs_state=None))
    layout = TapeLayout(GEOMETRY, 5)
    state = torch.ones(8, model.config.channels, layout.height, layout.width)
    torch.manual_seed(12)
    first = model(state, 4)
    second = model(state, 4)
    assert set(torch.unique(first).tolist()) <= {0.0625, 0.125, 0.25, 0.5, 1.0}
    assert not torch.equal(first, second)
    expected_program = state[:, :1].unsqueeze(1).expand(-1, 5, -1, -1, -1)
    assert torch.equal(first[:, :, :1], expected_program)


def test_batched_fire_masks_match_repeated_step_sampling():
    model = make_model(ModelConfig(state_leak=0.2, fire_rate=0.6, max_abs_state=None))
    layout = TapeLayout(GEOMETRY, 4)
    state = torch.ones(3, model.config.channels, layout.height, layout.width)
    torch.manual_seed(4)
    rollout = model(state, 5)
    torch.manual_seed(4)
    repeated = [state]
    for _ in range(5):
        repeated.append(model.step(repeated[-1]))
    assert torch.equal(rollout, torch.stack(repeated, dim=1))


def test_softplus_is_available_as_hidden_activation():
    relu = UpdateRule(ModelConfig(activation="relu"), 1)
    softplus = UpdateRule(ModelConfig(activation="softplus"), 1)
    with torch.no_grad():
        for rule in (relu, softplus):
            rule.hidden.weight.zero_()
            rule.hidden.bias.fill_(-1.0)
            rule.output.weight.fill_(1.0)
    perception = torch.zeros(1, 1, 1, 1)
    assert torch.count_nonzero(relu(perception)) == 0
    assert torch.all(softplus(perception) > 0)


@pytest.mark.parametrize("wrap_y", [False, True])
def test_optimized_padding_matches_explicit_padding(wrap_y):
    perception = Perception(ModelConfig(), wrap_y=wrap_y)
    state = torch.randn(2, perception.channels, 3, 9)
    filters = perception.filters()
    if wrap_y:
        padded = F.pad(state, (0, 0, 1, 1), mode="circular")
        padded = F.pad(padded, (1, 1, 0, 0), mode="constant")
    else:
        padded = F.pad(state, (1, 1, 1, 1), mode="constant")
    expected = F.conv2d(padded, filters, groups=perception.channels)
    assert torch.equal(perception(state, filters), expected)


def test_vertical_wrap_never_wraps_horizontally():
    config = ModelConfig(fixed_kernels=("sobel_y",), learnable_kernels=0, hidden_size=2)
    wrapped = Perception(config, wrap_y=True)
    zero = Perception(config, wrap_y=False)
    state = torch.zeros(1, config.channels, 3, 5)
    state[:, :, 0, 2] = 1.0
    assert torch.count_nonzero(wrapped(state)[:, :, 2, 2]) > 0
    assert torch.count_nonzero(zero(state)[:, :, 2, 2]) == 0
    state.zero_()
    state[:, :, 1, 0] = 1.0
    horizontal = Perception(
        ModelConfig(fixed_kernels=("sobel_x",), learnable_kernels=0), wrap_y=True
    )(state)
    assert torch.count_nonzero(horizontal[:, :, :, -1]) == 0


def test_noise_is_training_only_and_zero_avoids_randomness():
    model = make_model()
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(layout.render_tape(torch.ones(1, 3)), torch.tensor([0]))
    before = torch.get_rng_state()
    model.step(state, 0.0)
    assert torch.equal(before, torch.get_rng_state())
    model.eval()
    assert torch.equal(model.step(state, 0.2), model.step(state, 0.2))


def test_random_kernel_is_seeded_and_normalized():
    first = perception_kernel("random", 7)
    assert torch.equal(first, perception_kernel("random", 7))
    assert float(first.norm()) == pytest.approx(1.0)


@pytest.mark.parametrize("noise", [-0.1, float("nan"), float("inf"), True, "0"])
def test_step_and_rollout_reject_invalid_noise(noise):
    model = make_model()
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(
        layout.render_tape(torch.zeros(1, 3)), torch.tensor([0])
    )
    with pytest.raises(ValueError, match="perception_noise_std"):
        model.step(state, noise)
    with pytest.raises(ValueError, match="perception_noise_std"):
        model(state, 1, noise)


@pytest.mark.parametrize("invalid", ["channels", "geometry", "integer", "empty"])
def test_step_and_rollout_validate_state(invalid):
    model = make_model()
    layout = TapeLayout(GEOMETRY, 3)
    state = torch.zeros(1, model.config.channels, layout.height, layout.width)
    if invalid == "channels":
        state = state[:, :1]
    elif invalid == "geometry":
        state = state[..., :-1]
    elif invalid == "integer":
        state = state.long()
    else:
        state = state[:0]
    with pytest.raises(ValueError):
        model.step(state)
    with pytest.raises(ValueError):
        model(state, 0)


def test_prepared_filters_preserve_rollout_outputs_and_gradients():
    config = ModelConfig(
        program_mode="learned_read_only", fire_rate=1.0, max_abs_state=None
    )
    rollout_model = make_model(config)
    step_model = make_model(config)
    with torch.no_grad():
        rollout_model.rule.output.weight.normal_(std=0.01)
    step_model.load_state_dict(rollout_model.state_dict())
    layout = TapeLayout(GEOMETRY, 4)
    inputs = layout.render_tape(torch.randn(2, 4))
    tasks = torch.tensor([0, 1])
    rollout = rollout_model(rollout_model.initial_state(inputs, tasks), 3)
    states = [step_model.initial_state(inputs, tasks)]
    for _ in range(3):
        states.append(step_model.step(states[-1]))
    reference = torch.stack(states, dim=1)
    torch.testing.assert_close(rollout, reference)
    rollout.square().mean().backward()
    reference.square().mean().backward()
    for (name, actual), (_, expected) in zip(
        rollout_model.named_parameters(), step_model.named_parameters()
    ):
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad)
