# NCPU-EM-2D

NCPU-EM-2D is a two-dimensional **neural cellular automaton (NCA)**, inspired by
[NCPU](https://ncpu.pages.dev/) and the paper
[*Emergent Models: Intelligence from Tiny Substrates*](https://arxiv.org/abs/2608.14019).
It is an active experimental part of the ongoing NCPU work.

A small neural network updates each grid cell from its local neighborhood.
Applying the same rule repeatedly lets the system transform binary strings
on a tape. Different tasks share the update network and use different learned
program tiles.

## Tape and interface

Data uses three symbols: binary `0`, binary `1`, and `B`.
`B` is a blank and can also separate values.

| Symbol | State value |
|---|---:|
| `0` | -1 |
| `1` | +1 |
| `B` | 0 |

An **interface** is the convention for supplying inputs and reading outputs.
Here, input initializes one channel at the logical tape positions; output is
read from those same positions in the same channel after evolution.
Other data types can be serialized externally into strings without changing
the model's layout. There is no learned encoder or decoder.

Every grid cell holds continuous **channels**: program channels, one shared
mutable input/output channel, and computation channels. State has shape
`[batch, channels, height, width]`.

The tape lies on the middle row, with configurable stride and equal borders
on opposite sides:

```text
height = 2 * vertical_space + 1
width  = 2 * horizontal_space + (tape_slots - 1) * stride + 1

. . . . . . . . . . .
. . T . T . T . T . .
. . . . . . . . . . .
```

Logical position `i` is at
`(vertical_space, horizontal_space + i * stride)`.
Horizontal space must be divisible by stride. Only logical tape positions
receive input and loss; other grid cells provide computation space.
Horizontal perception always uses zero padding. Vertical perception can use
zero padding or circular wrapping.

Input and output are left-aligned and must each fit within `tape_slots - 1`,
leaving at least one blank. MSE supervises the complete target tape,
including every remaining blank. Targets are never written into the evolving
state.

## Program and emergent computation

A **program** is learned, task-specific state, not source code or a hand-written
instruction list. Each task has a tile of shape
`[program_channels, grid_height, stride]`, repeated horizontally without overlap.
Its parameter count is independent of tape capacity.

`PROGRAM_START = 0` repeats from the first column, cropping the final tile if
needed. The default `1` leaves column zero initially blank and repeats from
column one. This offset does not change the tape positions or border sizes.

Programs can be:

- `zero`: fixed all-zero state, for single-task experiments;
- `learned_read_only`: learned tiles held fixed during evolution;
- `learned_mutable`: learned tiles that also evolve with the state.

Read-only programs still receive gradients through their influence on mutable
channels. In mutable mode, the leading zero column is only an initial condition.
Rule and program training flags and weight decays are independent.

An **emergent model**, here, performs the global transformation through repeated
local interactions. Training learns the rule and program rather than specifying
the intermediate algorithm. This describes how computation is organized; it
does not establish computational universality or reliable extrapolation.

## Local update

Shared fixed and learned 3×3 kernels extract neighborhood features separately
for each channel. A pointwise MLP produces a residual update:

```text
perception -> 1×1 hidden projection -> activation -> 1×1 output projection
```

Identity, horizontal/vertical Sobel, optional Laplacian, and trainable kernels
are supported. Learned kernels can start as Laplacians or seeded random filters.
The activation is ReLU or Softplus; optional gates multiply the learned delta.
There is no attention or state normalization.

For mutable channels:

```text
next_state = clip(state + fire_mask * (gated_delta - state_leak * state))
```

Leak uses the same firing mask as the learned delta. Clipping follows the update;
frozen program values are restored exactly. New configurations default to leak
`0.02`; the supplied notebook explicitly uses `0.1`. Set it to zero to disable it.

Perception filters and fresh random firing masks are prepared once per trajectory.
They are not reused across rollouts. Optional perception noise applies only
during training and is linearly annealed across optimizer updates.

## Training

One optimizer update accumulates gradients from sequential trials at different
grid sizes, without padding them into one common shape. Each paired base case
receives independent uniform percentage variations for tape capacity and input
maximum, rounded with `floor(x + 0.5)`.

If a pair cannot fit both input and output with a blank, only the input maximum
is resampled; the sampled tape capacity is retained. Free evolution time scales
with the actual logical tape capacity, not its physical grid width:

```text
free_steps = round_half_up(steps_per_tape_slot * tape_slots * (1 + time_variation_draw))
supervision_steps = round_half_up(supervision_ratio * free_steps)
```

With `F` free steps and `S` supervised steps, loss applies at steps `F+1` through
`F+S`. Trial losses are averaged before the optimizer update.

The training domain includes every binary string through each trial's input
maximum, indexed as empty, `0`, `1`, `00`, `01`, `10`, `11`, and so on.
Batches sample uniformly over **strings, not lengths**. The empty input is an
all-blank tape. Tasks receive equal numbers of examples in a shuffled batch.

Task weights are configured as `TASKS = (("copy", 1.0), ("reverse", 2.0), ...)`.
Each task's MSE averages over examples, supervised steps, and logical positions:

```text
loss = sum(task_weight * task_mean_mse) / sum(task_weights)
```

Learning rates follow editable progress/rate anchors with linear or cosine
interpolation, allowing plateaus, decay, and deliberate increases.

Supported binary tasks are copy, bit_not, reverse, reverse_not, shift_left_zero,
shift_right_zero, gray_encode, prefix_xor, increment, parity, append_0, and append_1.
Shifts preserve width; increment wraps modulo `2^input_length`.
Empty input remains empty for length-preserving tasks, produces `0` for parity,
and produces the appended bit for append tasks.

**Dataset size grows exponentially with input length.** This 2D version
materializes each trial's complete input/target population before sampling its
batch. Evaluation limits reduce rollout work, not dataset construction.
The supplied sizes are modest; large input maxima can require substantial RAM.

## Run

Dependencies are Python 3.10–3.12, PyTorch, and Pillow. The notebook also uses
JupyterLab, IPykernel, and Matplotlib; development checks use pytest, Black, and
Flake8. Versions are declared in `pyproject.toml`.

From the repository folder:

```text
python -m pip install -e ".[notebook,dev]"
python -m jupyterlab notebooks/run.ipynb
```

The Python import remains `ncpu_computer`. Use a separate kernel/environment
when working with another project providing that same package name.

The single notebook has five cells:

1. **Configuration:** tasks, model, training settings, explicit test cases,
   mandatory structural checks, and printed layouts.
2. **Training/loading:** training, checkpoint loading or resume, per-task
   validation curves, running mean loss, and learning rate.
3. **Evaluation:** the fixed post-training cases.
4. **Visualization:** optional I/O evolution GIF with the initial program tile
   displayed separately; generated files go to `outputs/`.
5. **Programs:** plots of the learned task tiles.

The supplied settings retain the current experiment: seven tasks, reverse and
reverse_not weighted twice, 2,000 optimizer updates, 24 examples per task per
trial, five base pairs, hidden size 96, ReLU, and learned read-only programs.
They are experiment settings, not a claim of optimality. All scientific
hyperparameters remain editable in the first cell; execution controls are in
their respective cells.

## Evaluation and checkpoints

Periodic validation covers all strings through the base input maxima, using
fixed base geometries and times. Weighted validation MSE selects `best.pt`.

Final cases instead evaluate strings of **exactly their specified input length**,
including leading zeros. The supplied cases test the large base setup,
a longer tape with unchanged input length, and a longer tape with longer inputs.
Invalid capacities raise errors; inputs and outputs are never truncated or
resampled in these tests.

`EVALUATION_MAX_EXAMPLES = None` retains exhaustive evaluation.
An explicit positive limit selects a reproducible uniform subset.
Readout uses the literal strict thresholds `+0.333` and `-0.333`:
above is `1`, below is `0`, and the middle interval is `B`.
Single-output interpretation stops at the first blank; multiple-output
interpretation stops at two consecutive blanks.

- **Semantic:** correct output and required terminator, ignoring the later tail.
- **Raw:** every tape symbol matches, including blanks.
- **Symbol:** fraction of matching tape positions.
- **Stable:** semantic correctness throughout the evaluation window.

MSE and semantic/raw/symbol scores average over the configured window.
Task aggregates are unweighted. Reports therefore distinguish local symbol
accuracy from complete-string correctness; easy tasks can hide poor reversal
in aggregate scores.

Checkpoints are local under `checkpoints/<experiment>/`. `latest.pt` supports
exact resume, including optimizer and random-generator states.
Resume requires matching configuration and task weights; requesting a missing
checkpoint raises an error. Preserve checkpoints before reusing an experiment
name. Checkpoints are never included in the repository.

Format 8 is current. Format 7 loads with unit task weights, ReLU, and zero leak,
preserving its original dynamics. Load only checkpoints you trust.

## Checks and scope

`src/` contains the implementation, `notebooks/` the workflow, and `tests/`
focused regression tests:

```text
python -m pytest -q
```

Tests cover geometry, task semantics, masked dynamics, gradients, weighted loss,
trial sampling, accumulation, checkpoint resume, and tiny notebook runs.
No expensive training is required.

This repository preserves the convolution-only 2D design. Attention, the 1D
architecture, and the planned Weights & Biases runner are not included.
Reversal extrapolation remains experimental; performance depends on training
setup and is not guaranteed.
