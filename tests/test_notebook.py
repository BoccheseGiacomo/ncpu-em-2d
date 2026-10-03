import ast
import json
from pathlib import Path

import pytest

from ncpu_computer import TestCase


PROJECT_ROOT = Path(__file__).parents[1]


def notebook_cells():
    notebook = json.loads(
        (PROJECT_ROOT / "notebooks" / "run.ipynb").read_text(encoding="utf-8")
    )
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 5
    for cell in cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        compile("".join(cell["source"]), "notebook", "exec")
    return ["".join(cell["source"]) for cell in cells]


def execute_cell(source, namespace, overrides=None):
    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in (overrides or {}):
                statement.value = ast.parse(
                    repr(overrides[target.id]), mode="eval"
                ).body
    exec(compile(ast.fix_missing_locations(tree), "notebook", "exec"), namespace)


def test_single_notebook_exposes_complete_workflow():
    assert list((PROJECT_ROOT / "notebooks").glob("*.ipynb")) == [
        PROJECT_ROOT / "notebooks" / "run.ipynb"
    ]
    cells = notebook_cells()
    for setting in (
        "TASKS",
        "ACTIVATION",
        "STATE_LEAK",
        "PROGRAM_START",
        "BASE_TAPE_SLOTS",
        "BASE_INPUT_MAX_LENGTHS",
        "LR_POINTS",
        "TEST_CASES",
        "EVALUATION_BATCH_SIZE",
        "EVALUATION_MAX_EXAMPLES",
    ):
        assert f"{setting} =" in cells[0]
    assert "validate_experiment(config, TASK_NAMES)" in cells[0]
    assert "layout.schema()" in cells[0]
    assert "RUN_" not in cells[0]
    assert "RUN_TRAINING =" in cells[1]
    assert "RUN_EVALUATION =" in cells[2]
    assert "RUN_VISUALIZATION =" in cells[3]
    assert "metrics.validation_accuracies" in cells[1]
    assert 'loss_axis.set_yscale("log")' in cells[1]
    assert 'learning_rate_axis.set_yscale("log")' in cells[1]
    assert "model.programs.detach()" in cells[4]


@pytest.fixture
def notebook_namespace(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("IPython")
    matplotlib.use("Agg", force=True)
    monkeypatch.chdir(PROJECT_ROOT)
    cells = notebook_cells()
    namespace = {"TestCase": TestCase}
    execute_cell(
        cells[0],
        namespace,
        {
            "TASKS": (("copy", 1.0), ("reverse", 2.0)),
            "PROGRAM_CHANNELS": 1,
            "COMPUTATION_CHANNELS": 1,
            "HIDDEN_SIZE": 4,
            "FIRE_RATE": 0.9,
            "UPDATES": 2,
            "BATCH_SIZE_PER_TASK": 2,
            "BASE_TAPE_SLOTS": (3, 4),
            "BASE_INPUT_MAX_LENGTHS": (1, 2),
            "TAPE_VARIATION": 0.0,
            "INPUT_VARIATION": 0.0,
            "FREE_STEPS_PER_TAPE_SLOT": 0.5,
            "TIME_VARIATION": 0.0,
            "SUPERVISION_RATIO": 1.0,
            "LR_POINTS": ((0.0, 0.01), (1.0, 0.005)),
            "DEVICE": "cpu",
            "VALIDATION_EVERY": 1,
            "CHECKPOINT_EVERY": 1,
            "TEST_CASES": (TestCase("tiny", 4, 2, 1, 2),),
            "EVALUATION_BATCH_SIZE": 2,
            "EVALUATION_MAX_EXAMPLES": None,
        },
    )
    namespace["checkpoint_dir"] = tmp_path / "checkpoints"
    namespace["checkpoint_path"] = namespace["checkpoint_dir"] / "best.pt"

    class Display:
        def update(self, value):
            pass

    namespace["display"] = lambda *args, **kwargs: Display()
    return cells, namespace


def test_tiny_notebook_training_evaluation_and_program_plots(notebook_namespace):
    cells, namespace = notebook_namespace
    execute_cell(cells[1], namespace, {"PROGRESS_EVERY": 1})
    assert namespace["trainer"].current_update == 2
    execute_cell(cells[2], namespace)
    assert {result.task for result in namespace["evaluation_results"]} == {
        "copy",
        "reverse",
        "aggregate",
    }
    execute_cell(cells[3], namespace, {"RUN_VISUALIZATION": False})
    execute_cell(cells[4], namespace)
    execute_cell(
        cells[1],
        namespace,
        {
            "RUN_TRAINING": False,
            "LOAD_CHECKPOINT": True,
        },
    )
    assert namespace["model"].task_names == ("copy", "reverse")


def test_notebook_missing_resume_is_an_error(notebook_namespace):
    cells, namespace = notebook_namespace
    with pytest.raises(FileNotFoundError):
        execute_cell(cells[1], namespace, {"RESUME_LATEST": True})


def test_visualization_readout_uses_the_displayed_trajectory(
    notebook_namespace, tmp_path
):
    cells, namespace = notebook_namespace
    execute_cell(
        cells[1],
        namespace,
        {
            "RUN_TRAINING": False,
            "LOAD_CHECKPOINT": False,
        },
    )
    captured = {}
    real_interpreter = namespace["interpret_tape"]

    def interpret(values, mode):
        captured["values"] = values.clone()
        return real_interpreter(values, mode)

    def save(trajectory, path, **kwargs):
        captured["trajectory"] = trajectory
        return path

    namespace["interpret_tape"] = interpret
    namespace["save_gif"] = save
    namespace["DisplayImage"] = lambda **kwargs: None
    execute_cell(
        cells[3],
        namespace,
        {
            "VISUALIZATION_TASK": "reverse",
            "VISUALIZATION_INPUT": "10",
            "VISUALIZATION_TAPE_SLOTS": 4,
            "VISUALIZATION_FREE_STEPS": 1,
            "VISUALIZATION_SUPERVISION_STEPS": 2,
            "GIF_PATH": str(tmp_path / "preview.gif"),
        },
    )
    layout = namespace["layout"]
    model = namespace["model"]
    expected = layout.extract_tape(captured["trajectory"][-1, model.config.io_channel])
    namespace["torch"].testing.assert_close(captured["values"], expected.cpu())
