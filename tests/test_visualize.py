import pytest
import torch
from PIL import Image

from ncpu_computer import (
    ExperimentConfig,
    NeuralCellularAutomaton,
    TapeLayout,
    evolution_phase,
    rollout_rgb,
    save_gif,
)


def test_evolution_phase_uses_explicit_trial_times():
    assert evolution_phase(1, 1, 2) == "Free evolution"
    assert evolution_phase(2, 1, 2) == "Supervision window"
    assert evolution_phase(4, 1, 2) == "Beyond training window"
    with pytest.raises(ValueError):
        evolution_phase(-1, 1, 2)


def test_rollout_rgb_shows_only_shared_io_on_fixed_scale():
    rollout = torch.zeros(1, 4, 1, 5)
    rollout[0, 1, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    rgb = rollout_rgb(rollout, 1)
    assert rgb.shape == (1, 1, 5, 3)
    assert torch.equal(rgb[0, 0, 0], rgb[0, 0, 1])
    assert torch.equal(rgb[0, 0, 3], rgb[0, 0, 4])
    assert torch.equal(rgb[0, 0, 2], torch.tensor([245, 245, 245]))


def test_save_gif_includes_all_frames_and_program_tile(tmp_path):
    config = ExperimentConfig()
    layout = TapeLayout(config.geometry, 3)
    model = NeuralCellularAutomaton(config.model, config.geometry, ("copy",))
    inputs = layout.render_tape(torch.tensor([[1.0, -1.0, 0.0]]))
    initial = model.initial_state(inputs, torch.tensor([0]))
    rollout = model(initial, 2)[0]
    path = save_gif(
        rollout,
        tmp_path / "evolution.gif",
        layout=layout,
        config=config,
        program_tile=model.programs[0],
        task_name="copy",
        input_symbols="10",
        target_symbols="10",
        free_steps=1,
        supervision_steps=1,
        duration_ms=20,
        scale=4,
    )
    with Image.open(path) as image:
        assert image.n_frames == 3
