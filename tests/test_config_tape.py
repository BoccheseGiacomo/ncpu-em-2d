from dataclasses import replace

import pytest
import torch

from ncpu_computer import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TapeLayout,
    TestCase,
    TERNARY_THRESHOLD,
    binary_to_integer,
    encode_strings,
    integer_to_binary,
    interpret_tape,
    quantize,
    round_half_up,
    tensor_to_symbols,
    varied_bounds,
)


def test_config_round_trip_and_channel_roles():
    config = ExperimentConfig(
        geometry=GeometryConfig(stride=3, vertical_space=2, horizontal_space=6),
        model=ModelConfig(program_channels=2, computation_channels=4),
        test_cases=(TestCase("test", 8, 4, 3, 5),),
    )
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert config.model.io_channel == 2
    assert config.model.channels == 7
    assert config.geometry.height == 5
    with pytest.raises(ValueError, match="divisible"):
        replace(config.geometry, horizontal_space=2).validate()
    with pytest.raises(ValueError, match="zero program"):
        replace(
            config,
            training=replace(config.training, train_program=True),
        ).validate()
    with pytest.raises(ValueError, match="one blank"):
        replace(
            config,
            training=replace(
                config.training,
                base_tape_slots=(3, 6),
                base_input_max_lengths=(4, 5),
                tape_variation=0.0,
                input_variation=0.0,
            ),
        ).validate()
    with pytest.raises(ValueError, match="no feasible"):
        replace(
            config,
            training=replace(
                config.training,
                base_tape_slots=(5,),
                base_input_max_lengths=(4,),
                tape_variation=0.4,
                input_variation=0.0,
            ),
        ).validate()


def test_legacy_model_config_preserves_old_dynamics():
    config = ExperimentConfig()
    data = config.to_dict()
    del data["model"]["activation"]
    del data["model"]["state_leak"]
    loaded = ExperimentConfig.from_dict(data)
    assert loaded.model.activation == "relu"
    assert loaded.model.state_leak == 0.0


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("activation", "gelu", "activation"),
        ("state_leak", -0.01, "state_leak"),
        ("state_leak", 1.01, "state_leak"),
    ],
)
def test_model_rejects_invalid_dynamics(field, value, match):
    config = replace(ModelConfig(), **{field: value})
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_round_half_up_and_varied_bounds_are_explicit():
    assert [round_half_up(value) for value in (0.0, 1.49, 1.5, 2.5)] == [0, 1, 2, 3]
    assert varied_bounds(9, 0.3) == (6, 12)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"lr_interpolation": "cubic"}, "lr_interpolation"),
        ({"lr_points": ((0.1, 1e-3), (1.0, 1e-4))}, "start"),
        (
            {"lr_points": ((0.0, 1e-3), (0.5, 5e-4), (0.5, 1e-4), (1.0, 1e-4))},
            "strictly increase",
        ),
        ({"lr_points": ((0.0, 1e-3), (1.0, 0.0))}, "positive"),
    ],
)
def test_learning_rate_schedule_validation(changes, match):
    config = ExperimentConfig()
    with pytest.raises(ValueError, match=match):
        replace(config, training=replace(config.training, **changes)).validate()


def test_layout_has_exact_symmetric_edges():
    geometry = GeometryConfig(stride=2, vertical_space=1, horizontal_space=2)
    layout = TapeLayout(geometry, 4)
    values = torch.tensor([[1.0, -1.0, 0.0, 1.0]])
    grid = layout.render_tape(values)
    assert grid.shape == (1, 3, 11)
    assert layout.tape_coordinates == ((1, 2), (1, 4), (1, 6), (1, 8))
    assert layout.tape_coordinates[0][1] == 2
    assert layout.width - 1 - layout.tape_coordinates[-1][1] == 2
    assert torch.equal(layout.extract_tape(grid), values)
    assert torch.count_nonzero(grid) == 3


def test_direct_ternary_codec_and_interpreter():
    encoded = encode_strings(("10B", "", "B01"), tape_slots=4)
    assert encoded.tolist() == [
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
    ]
    values = torch.tensor(
        [
            -TERNARY_THRESHOLD - 1e-6,
            -TERNARY_THRESHOLD,
            TERNARY_THRESHOLD,
            TERNARY_THRESHOLD + 1e-6,
        ]
    )
    assert quantize(values).tolist() == [-1, 0, 0, 1]
    assert tensor_to_symbols(encoded[0]) == "10BB"
    interpreted = interpret_tape(encoded[0])
    assert interpreted.binary_strings == ("10",)
    assert interpreted.terminated


def test_integer_codec_is_minimal_binary():
    assert integer_to_binary(0) == "0"
    assert integer_to_binary(11) == "1011"
    assert binary_to_integer("1011") == 11
    with pytest.raises(ValueError, match="minimal"):
        binary_to_integer("00")
