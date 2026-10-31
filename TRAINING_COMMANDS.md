# Training commands

Here are the commands you can use to train a new model from scratch, depending on which training paradigm you prefer.

> [!TIP]
> Ensure your virtual environment is activated before running any training:
> ```bash
> source .venv/bin/activate
> ```

---

### 1. Vector RL Training (Recommended — Fastest & Modern)

This uses the Numba-accelerated parallel vector engine ([`magic_pong/ai/vector_training.py`](file:///home/adrien/Work/Perso/magic_pong/magic_pong/ai/vector_training.py)), simulating 128 games concurrently at **50,000–65,000 frames/second** on CPU. A full 40-million frame run trains from zero to competitive mastery in **~10–12 minutes**.

```bash
python -m magic_pong.ai.vector_training \
    --output models/my_vector_dqn \
    --total_frames 40000000 \
    --geometry_context \
    --dueling \
    --final_panel
```

* **`--geometry_context`**: Enables Schema v7 (16 contact geometry and multi-bounce features projected into the first hidden layer).
* **`--dueling`**: Enables Dueling DQN ($V(s) + A(s, a)$) with analytical fold-on-save.
* **`--final_panel`**: Automatically benchmarks the resulting `best.pth` checkpoint against the 108-match exact competitive panel upon completion.

#### Quick 10-Second Smoke Test
To verify the engine runs end-to-end on your machine before launching a long run:
```bash
python -m magic_pong.ai.vector_training \
    --output models/smoke_test \
    --total_frames 500000 \
    --geometry_context \
    --eval_interval_frames 250000
```

---

### 2. Planner Distillation (Supervised Imitation Learning)

If you want to train a model without RL exploration by directly distilling the Lookahead Planner's tree-search rollouts ([`magic_pong/ai/distillation.py`](file:///home/adrien/Work/Perso/magic_pong/magic_pong/ai/distillation.py)):

```bash
python -m magic_pong.ai.distillation \
    --labels 100000 \
    --output models/my_distilled_model.pth \
    --geometry_context \
    --evaluate
```

* **`--labels 100000`**: Collects 100,000 state-action rollouts from the Lookahead Planner.
* **`--evaluate`**: Evaluates the network against the full competitive panel at the end.

---

### 3. The Full Production Recipe (How `dqn_v7_promoted.pth` was Built)

To reproduce the exact sequence used to create the current production champion:

```bash
# Step 1: Distill initial tactical knowledge from the planner
python -m magic_pong.ai.distillation \
    --labels 100000 \
    --output models/stage1_distilled.pth \
    --geometry_context

# Step 2: Vector RL fine-tuning with Dueling heads and frozen policy anchor
python docs/ai/run_vector_rl_promoted_finetuning.py \
    --initial_checkpoint models/stage1_distilled.pth \
    --output_dir models/stage2_rl

# Step 3: Curvature & Bounce-Targeted DAgger (Round 3)
python docs/ai/run_curvature_dagger_round3.py \
    --base_model models/stage2_rl/best.pth \
    --output_dir models/my_champion
```

---

### 4. Traditional Frame Pipeline (`train_optimized.py`)

If you want to use the legacy episode-based curriculum pipeline ([`train_optimized.py`](file:///home/adrien/Work/Perso/magic_pong/train_optimized.py)) through progressive phases (curriculum $\to$ annealing $\to$ self-play league):

```bash
python train_optimized.py \
    --mode frame_pipeline \
    --output_dir models/my_pipeline_run \
    --frame_geometry_context \
    --pipeline_curriculum_frames 2000000 \
    --pipeline_anneal_frames 2000000 \
    --pipeline_selfplay_frames 2000000
```

---
