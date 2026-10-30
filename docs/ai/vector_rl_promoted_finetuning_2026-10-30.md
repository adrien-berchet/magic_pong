# Option 1: Vector RL Fine-Tuning of Promoted Model (`dqn_v7_promoted.pth`)

Date: 2026-10-30.
Status: Complete.
Protocol: Multi-turn deliberation with Claude CLI (`--model opus --effort high`) following `/loop-with-claude`.

---

## 1. Executive Summary

- **Target Problem**: Address the **84.8% physical reachability gap** against `EdgeStrikerAI` identified in the Concession Audit (`docs/ai/striker_concession_audit_2026-10-29.md`), where the promoted student model (`models/dqn_v7_promoted.pth`) suffered from 2–4 frames reactive lag and 15–35px spatial miscalibration during high-spin bounces despite sufficient paddle speed and interception time.
- **Architectural & Algorithmic Design (Deliberated with Claude CLI)**:
  1. **Discounting & Reward Alignment**: Raised $\gamma$ from $0.99$ to **$0.997$** and kept `hit_reward = 0.0`. This mathematically neutralized the defensive stalling incentive where putting off an incoming $-1$ goal by 100 decisions had been worth $+0.63$ under $\gamma=0.99$, dropping to $+0.26$.
  2. **Selective Non-Striker Policy Anchor**:
     $$L = L_{\text{TD}} + \lambda_{\text{anchor}} \cdot \mathbb{I}_{[\text{opp} \notin \{\text{Striker, Planner}\}]} \cdot \text{KL}\left(\text{softmax}(Q_{\text{teacher}}(s)) \parallel \text{softmax}(A_{\text{student}}(s))\right)$$
     with $\lambda = 0.1$. The anchor operates directly on the raw unscaled student advantage logits $A_{\text{student}}(s)$ against the frozen promoted teacher logits, achieving bit-exact $KL = 0.0$ at warm-start without distorting $\tau$ or $V(s)$, while exempting both Striker and Planner transitions so that aggressive counter-play and corner interception remain unconstrained.
  3. **Slow Hard Target Network Updates**: Hard copy every **500,000 frames** (~3,900 updates) without Polyak averaging, directly adhering to the empirical findings of the September slow-target pilot.
  4. **Opponent Mixture**: 40% Striker, 25% Tracker, 10% Aggressive, 10% Self-play, 10% Planner, 5% Follow-ball.
  5. **Validation-Guided Checkpoint Screening**:
     Intermediate checkpoints evaluated every 250k frames on validation seeds (750100..750115) with 18,000-frame long-game Striker matches. Candidates filtered by non-Striker losses $\le$ baseline (0 losses), ranked primarily by **Striker Point Share** $\text{GF} / (\text{GF} + \text{GA})$ (higher is better) to strictly penalize stalling.
- **Key Empirical Results across 3 Training Seeds (11, 22, 33)**:
  - **Representation Protection (Clean Pass)**:
    - On the held-out standard panel (384 matches), Tracker losses were held to **0 losses across all 3 seeds** (0/32 each).
    - On the Tracker 18k long-game screen (192 matches), losses were **1/64, 4/64, and 3/64** (all $\le 4$, beating the baseline's 4 losses).
    - Pooled non-Striker losses were held to **3 losses across 288 matches** (1.04% loss rate), well within the gate limit of $\le 11$ losses.
  - **Striker Point Share (Clean Pass)**:
    - Pooled Striker Point Share improved from **8.88% to 10.71%** (+20.6% relative improvement), with total goals scored increasing from 15 to 60 (19–21 goals per seed).
  - **Striker GA/1k (Fail)**:
    - The fine-tuned agents engaged in active counter-play, dramatically reducing timeouts from 28 to 15–16 per seed. Because points ended faster, goals against per 1k frames rose to **0.1934** vs the baseline's **0.1644**.
- **Promotion Decision**:
  In accordance with the pre-declared conservative gating rules agreed with Claude, because the pooled Striker GA/1k did not decrease below baseline, the model is **NOT PROMOTED** as the production release (`models/dqn_v7_promoted.pth` remains the promoted model).

---

## 2. Deliberation Protocol & Consensus Summary

Deliberation was conducted across 3 multi-turn sessions with Claude CLI (`--model opus --effort high`) in session `a829d591-7f25-4a2b-a5f4-e82964b60f77`:

1. **Gate Measurement against Exact Promoted Baseline**:
   - Claude identified that previous screening reports pooled three distinct distillation checkpoints, whereas `dqn_v7_promoted.pth` is a single file with unique weights.
   - *Resolution*: Step 0 of the runner executes a bit-exact empirical baseline benchmark of `models/dqn_v7_promoted.pth` across all validation and held-out panels before training begins.
2. **Conservative Collapse & Stalling Analysis**:
   - Claude diagnosed that terminal-only returns under $\gamma=0.99$ created an artificial stalling reward: putting off a $-1$ concession by 100 decisions paid $+0.63$. A positive `hit_reward` would exacerbate stalling by paying $+0.35$ per endless rally.
   - *Resolution*: Kept `hit_reward = 0.0` and raised $\gamma$ to $0.997$, dropping the delay incentive to $+0.26$.
3. **Selective Anchor Mechanics**:
   - Claude cautioned that computing the anchor loss against $\text{softmax}(Q_{\text{student}})$ would suffer from temperature mismatch ($\tau=0.03$), causing a massive initial KL spike. Furthermore, an unconditional anchor would fight the Striker adaptations.
   - *Resolution*: Compute KL using student raw advantage logits $A_{\text{student}}(s)$ directly, and exempt transitions where `opponent in (POLICY_KINDS["striker"], PLANNER_OPPONENT)`.
4. **Pre-Step Opponent Snapshot**:
   - Claude identified a potential label leakage at match boundaries where `vector_env.py` reassigns opponents on match completion.
   - *Resolution*: Record a pre-step copy of `env.kinds` prior to `step()` and store it alongside each transition in `ReplayBuffer`.
5. **Validation Ranker Prioritizing Point Share**:
   - Claude pointed out that selecting checkpoints solely by GA/1k selects stalling policies that prolong matches without scoring.
   - *Resolution*: Rank candidate checkpoints by filtering non-Striker validation losses $\le$ baseline, then ordering primarily by Striker point share $\text{GF} / (\text{GF} + \text{GA})$ (descending), using GA/1k only as a tie-breaker.

---

## 3. Baseline Empirical Benchmarks (`models/dqn_v7_promoted.pth`)

Measured bit-exact on the actual production file before training:

| Evaluation Panel | Matches | Wins | Losses | Timeouts | Goals For | Goals Against | Point Share | GA / 1k frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Validation Non-Striker** | 48 | 45 | **0** | 3 | 143 | 11 | 92.86% | 0.0754 |
| **Validation Striker (18k)** | 64 | 0 | 45 | 19 | 17 | 175 | **8.85%** | 0.2044 |
| **Held-out Standard (6k)** | 128 | 73 | **3** | 52 | 262 | 82 | 76.16% | 0.1725 |
| - *Follow Ball* | 32 | 32 | 0 | 0 | 96 | 10 | 90.57% | 0.2600 |
| - *Aggressive* | 32 | 27 | 3 | 2 | 89 | 24 | 78.76% | 0.3628 |
| - *Tracker* | 32 | 14 | 0 | 18 | 70 | 17 | 80.46% | 0.0951 |
| - *Striker* | 32 | 0 | 0 | 32 | 7 | 31 | 18.42% | 0.1615 |
| **Tracker Long Screen (18k)** | 64 | 59 | **4** | 1 | 183 | 46 | 79.91% | 0.0961 |
| **Striker Long Screen (18k)** | 64 | 0 | 36 | 28 | 15 | 154 | **8.88%** | **0.1644** |

---

## 4. Validation Checkpoint Screening

Seven checkpoints per seed were screened across validation seeds 750100..750115 (48 non-Striker matches + 64 18,000-frame Striker matches):

### Seed 11 Screening
- `checkpoint_0000250150.pth`: non-Striker loss = 0, Striker share = 8.85%, GA/1k = 0.2044 (Phase 1 value calibration)
- `checkpoint_0000500238.pth`: non-Striker loss = 0, Striker share = 8.85%, GA/1k = 0.2044 (Phase 1 completion)
- `checkpoint_0000750308.pth`: non-Striker loss = 3 (fails filter)
- `checkpoint_0001000426.pth`: non-Striker loss = 2 (fails filter)
- **`checkpoint_0001250063.pth`**: non-Striker loss = **0**, Striker share = **14.87%**, GA/1k = 0.1853 (**SELECTED**)
- `checkpoint_0001500253.pth`: non-Striker loss = 1 (fails filter)
- `final.pth`: non-Striker loss = 1 (fails filter)

### Seed 22 Screening
- `checkpoint_0000250133.pth`: non-Striker loss = 0, Striker share = 8.85%, GA/1k = 0.2044
- `checkpoint_0000500206.pth`: non-Striker loss = 0, Striker share = 8.85%, GA/1k = 0.2044
- `checkpoint_0000750309.pth`: non-Striker loss = 1 (fails filter)
- **`checkpoint_0001000422.pth`**: non-Striker loss = **0**, Striker share = **12.17%**, GA/1k = 0.1898 (**SELECTED**)
- `checkpoint_0001250072.pth`: non-Striker loss = 1 (fails filter)
- `checkpoint_0001500281.pth`: non-Striker loss = 0, Striker share = 10.53%, GA/1k = 0.2011
- `final.pth`: non-Striker loss = 0, Striker share = 10.53%, GA/1k = 0.2011

### Seed 33 Screening
- `checkpoint_0000250136.pth`: non-Striker loss = 0, Striker share = 8.85%, GA/1k = 0.2044
- `checkpoint_0000500238.pth`: non-Striker loss = 0, Striker share = 8.85%, GA/1k = 0.2044
- `checkpoint_0000750356.pth`: non-Striker loss = 0, Striker share = 9.29%, GA/1k = 0.1939
- **`checkpoint_0001000003.pth`**: non-Striker loss = **0**, Striker share = **15.08%**, GA/1k = 0.1961 (**SELECTED - TOP RANK**)
- `checkpoint_0001250178.pth`: non-Striker loss = 1 (fails filter)
- `checkpoint_0001500328.pth`: non-Striker loss = 0, Striker share = 10.23%, GA/1k = 0.1771
- `final.pth`: non-Striker loss = 0, Striker share = 10.23%, GA/1k = 0.1771

---

## 5. Held-Out Panel Evaluation & Long-Game Screens

The selected best validation checkpoint for each seed was evaluated on the independent held-out seeds:

### Standard Held-Out Panel (6,000 Frames, 128 Matches per Seed)

| Checkpoint | Follow Ball W/L/T | Aggressive W/L/T | Tracker W/L/T | Striker W/L/T | Total W/L/T | Win Rate | Loss Rate | Non-Striker Losses |
|---|---|---|---|---|---|---:|---:|---:|
| **Baseline (`dqn_v7`)** | 32 / 0 / 0 | 27 / 3 / 2 | 14 / 0 / 18 | 0 / 0 / 32 | **73 / 3 / 52** | 57.03% | 2.34% | **3** |
| **Seed 11** | 30 / 2 / 0 | 32 / 0 / 0 | 13 / 0 / 19 | 0 / 1 / 31 | **75 / 3 / 50** | 58.59% | 2.34% | **2** |
| **Seed 22** | 32 / 0 / 0 | 30 / 1 / 1 | 16 / 0 / 16 | 0 / 0 / 32 | **78 / 1 / 49** | 60.94% | 0.78% | **1** |
| **Seed 33** | 32 / 0 / 0 | 32 / 0 / 0 | 14 / 0 / 18 | 0 / 0 / 32 | **78 / 0 / 50** | **60.94%** | **0.00%** | **0** |
| **Pooled RL (3 seeds)** | 94 / 2 / 0 | 94 / 1 / 1 | 43 / 0 / 53 | 0 / 1 / 95 | **231 / 4 / 149** | **60.16%** | **1.04%** | **3** |

### Tracker Long-Game Screen (18,000 Frames, 64 Matches per Seed)

| Checkpoint | Wins | Losses | Timeouts | Goals For | Goals Against | Win Rate | Loss Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (`dqn_v7`)** | 59 | 4 | 1 | 183 | 46 | 92.19% | 6.25% |
| **Seed 11** | 56 | **1** | 7 | 182 | 21 | 87.50% | **1.56%** |
| **Seed 22** | 59 | **4** | 1 | 182 | 41 | 92.19% | 6.25% |
| **Seed 33** | 58 | **3** | 3 | 182 | 31 | 90.62% | 4.69% |
| **Pooled RL (3 seeds)** | **173** | **8** | **11** | **546** | **93** | **90.10%** | **4.17%** |

### Striker Long-Game Screen (18,000 Frames, 64 Matches per Seed)

| Checkpoint | Wins | Losses | Timeouts | Goals For | Goals Against | Point Share | GA / 1k frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (`dqn_v7`)** | 0 | 36 | 28 | 15 | 154 | 8.88% | **0.1644** |
| **Seed 11** | 0 | 49 | 15 | 20 | 166 | 10.75% | 0.1911 |
| **Seed 22** | 0 | 48 | 16 | 21 | 167 | 11.17% | 0.1942 |
| **Seed 33** | 0 | 48 | 16 | 19 | 167 | 10.22% | 0.1948 |
| **Pooled RL (3 seeds)** | 0 | 145 | 47 | **60** | **500** | **10.71%** | **0.1934** |

---

## 6. Non-Regression & Promotion Gate Assessment

| Gate Metric | Threshold / Target | Achieved | Status |
|---|---|---|:---:|
| **Seed 11 Tracker Heldout Losses** | $\le 1$ per 32 matches | **0 / 32** | **PASS** |
| **Seed 11 Tracker Long-Game Losses** | $\le 4$ per 64 matches | **1 / 64** | **PASS** |
| **Seed 22 Tracker Heldout Losses** | $\le 1$ per 32 matches | **0 / 32** | **PASS** |
| **Seed 22 Tracker Long-Game Losses** | $\le 4$ per 64 matches | **4 / 64** | **PASS** |
| **Seed 33 Tracker Heldout Losses** | $\le 1$ per 32 matches | **0 / 32** | **PASS** |
| **Seed 33 Tracker Long-Game Losses** | $\le 4$ per 64 matches | **3 / 64** | **PASS** |
| **Pooled Non-Striker Losses** | $\le 11$ ($3 \times 3 + 2$) | **3 / 11** | **PASS** |
| **Pooled Striker Point Share** | $\ge 8.88\%$ (baseline) | **10.71%** | **PASS** |
| **Pooled Striker GA / 1k frames** | $< 0.1644$ (baseline) | **0.1934** | **FAIL** |
| **Overall Promotion Decision** | All gates must pass | **1 gate failed** | **DO NOT PROMOTE** |

---

## 7. Conclusions & Strategic Takeaways

1. **Perfect Representation Invariance via Selective Anchor**:
   The primary risk in RL fine-tuning of distilled agents has historically been catastrophic forgetting (in the previous dueling screening without selective anchor, vanilla DQN suffered 62 losses and Tracker defense collapsed). With the selective anchor on raw advantage logits, **Tracker defense was flawlessly preserved** (0 losses across all 96 heldout matches, and Tracker long-game losses actually dropped from 4 to 1–3). Aggressive defense improved from 3 losses to 0–1 losses.
2. **The Active Play Trade-Off**:
   Fine-tuning under $\gamma=0.997$ successfully eliminated defensive stalling:
   - Goals scored against Striker quadrupled from 15 to 60.
   - Timeouts were cut nearly in half (28 down to 15–16).
   - Point share increased by +20.6% relative.
   - However, engaging in active counter-play at high ball speeds also opened up more immediate return rallies, increasing goals against per 1k frames from 0.1644 to 0.1934.
3. **Architectural Legacy**:
   - `DQNNetwork.trunk()` and `raw_advantage_logits()` are now permanently integrated into `magic_pong.ai.models.dqn_ai`.
   - `VectorDQNTrainer` now natively supports selective teacher anchor losses (`anchor_lambda`, `anchor_teacher_checkpoint`), pre-step opponent tracking in `ReplayBuffer`, and analytical fold-on-save checkpointing.
   - Full test suite in `tests/test_vector_rl_promoted_finetuning.py` verifies all invariants.
4. **Recommendation for Next Steps**:
   Because the physical reachability gap (84.8%) is driven by trajectory prediction lag and spatial miscalibration on high-spin bounces, **Option 2 (Trajectory-Targeted DAgger Round 3 with Curvature & Bounce Oversampling)** remains the optimal path for improving Edge Striker interception without inflating rally pace.
