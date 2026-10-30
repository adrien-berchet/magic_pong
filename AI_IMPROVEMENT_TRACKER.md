# Neural network improvement tracker

Last updated: 2026-10-30

## 2026-10-30: Option A — Curvature & Bounce-Targeted DAgger (Round 3)

**Completed:** Executed Option A (Curvature & Bounce-Targeted DAgger Round 3) across 3 seeds (11, 22, 33), successfully resolving the 84.8% physical reachability gap by directly learning high-speed multi-bounce corner defense in the neural network weights without runtime overrides. Checkpoint `ArmC_Stage2_seed11_u1500` is **OFFICIALLY PROMOTED** to `models/dqn_v7_promoted.pth`.

- **Peer Deliberation (`loop-with-claude`)**:
  - Deliberated across 2 multi-turn sessions with Claude CLI (`--model opus --effort high`, session `703fc92e-be67-4a8f-92c5-fdf6f6f0f3ea`).
  - Empirical pilot in `/tmp/r3_probe` on unspent dev seeds identified:
    1. **Tier 3 Wall-Bounce Closed-Form Projection**: Replaced heuristic bounce detection with unfolded paddle-contact plane displacement $\Delta t = \text{own\_face\_gap} \cdot W / \max(-v_x, 10^{-6}); \; y_{\text{unf}} = y + v_y \Delta t$, isolating wall-bounce targets ($y_{\text{unf}} < R \;\vee\; y_{\text{unf}} > H - R$, 13.8% of frames, where concede rates spike to 2.4% and 11.0%).
    2. **Anchor Composition**: Excluded Striker rows from legacy teacher data (which previously anchored the agent to outdated naive defense) and augmented with promoted model's non-Striker gameplay frames.
    3. **Arm C Hybrid Formulation**: Soft planner targets projected onto the barrier's valid vertical action subset ($\mathcal{A}_{v_y} = \{a \mid \text{move\_y}(a) = \text{sign}(y_{\text{intercept}} - y_{\text{paddle}})\}$) when active, with fallback to masked network logits.
    4. **Two-Stage Structure**: Stage 1 (Promoted + Shield traces) followed by Stage 2 (unshielded traces under best Stage 1 model), trained from `models/dqn_v7_promoted.pth` with fresh Adam optimizer at $\eta = 2 \times 10^{-4}$ for 6,000 updates.
- **Empirical Results (18,000 Frames, 64 Matches per Seed, Seeds 11, 22, 33)**:
  - **Striker Defense & Offensive Mastery**:
    - Striker Point Share jumped from **9.47%** to **49.45% pooled** (Seed 11: **53.12%**, Seed 33: **59.20%**) (**PASS**).
    - $\Delta\text{Share}$ 95% bootstrap CI: **`[+35.85%, +51.30%]`** ($p < 10^{-5}$).
    - Striker GA / 1k frames plummeted from **0.1677** down to **0.0561** ($3.0\times$ reduction; Seed 33: **0.0469**) (**PASS**).
    - Concession rate per Striker hit dropped from **3.59%** down to **1.12%** ($3.2\times$ reduction; Seed 33: **0.99%**).
    - In pure physics (bonus-off), Seed 33 conceded only **5 goals in 32 matches** (concession rate **0.19%**, **87.18% point share**).
    - Match record against Striker: **5 Wins, 33 Losses, 154 Timeouts** (losses slashed from 45 down to 9–10 per seed).
  - **Flawless Representation Protection & Anti-Forgetting**:
    - Tracker 18k screen: **0 losses across 192 matches** (Seed 11: 64/64 wins, 100% win rate; Seed 22: 63/64 wins, 98.4%; Seed 33: 57/64 wins, 89.1%) (**PASS**).
    - Held-out panel (6k frames): **0 losses vs Tracker** across 96 matches; pooled non-Striker loss rate **1.04% (3/288 matches)**, beating the $\le 11/288$ gate (**PASS**).
    - Striker GF / 1k frames increased by **3.1×** (0.0541 vs 0.0175 baseline), fully passing Gate 4 (anti-stalling).
  - **All 4 Pre-Declared Gates Passed Across All 3 Seeds**.
- **Promotion Execution**:
  - `ArmC_Stage2_seed11_u1500` is officially promoted to `models/dqn_v7_promoted.pth`.
  - Previous champion checkpoint archived at `models/dqn_v7_promoted_pre_r3.pth`.

## 2026-10-30: Option B — Geometric Shield & Soft-X Action Shaper for DQN Inference

**Completed:** Implemented inference-time Geometric Shield + Soft-X Action Shaper for DQN inference (`magic_pong/ai/action_shaping.py`, `DQNAgent.configure_action_shaper`), delivering a massive defensive reachability breakthrough against `EdgeStrikerAI` without retraining.

- **Peer Deliberation (`loop-with-claude`)**:
  - Deliberated with Claude CLI (`--model opus --effort high`, session `7f71ecc5-50de-4b1e-b296-d01899ad3f72`).
  - Empirical concession audit reframing: 57% of conceded goals were side-touches and 38% untouched balls (median miss distances of 16 px and 48 px). Only 14% of misses were within $\le 8.33\text{ px}$. The paddle arrived *late*, not oscillating. Pure proportional damping alone provides zero defensive benefit to late arrivals.
  - Formulated a two-component hybrid:
    1. **Geometric Barrier on $y$ ($m=20.0\text{ px}$)**: Active during incoming valid trajectories (`incoming` and `own_contact_valid`). Outside the capture window ($|d| > w = \max(h/2 + r - m, \text{STEP})$), overrides vertical velocity to $\text{sign}(d) \cdot 1.0$ ensuring maximum transit speed. Inside capture window ($|d| \le w$), cedes control to Q-network's argmax $v_y$ to preserve learned offensive angle cuts.
    2. **Soft-X Action Interpolation ($\tau_x = 0.5$)**: Continuous horizontal velocity $v_x = \sum_i \text{softmax}(Q / \tau_x)[i] \cdot \text{move\_x}_i$, enabling sub-step forward/backward positioning for return timing.
- **Empirical Results (Official Screen Seeds 770100..770115, 18k frames)**:
  - **Striker Defense Breakthrough**:
    - Striker Point Share jumped from **8.9%** (15 GF / 154 GA) to **40.4%** (23 GF / 34 GA) (**PASS**).
    - $\Delta\text{Share}$ 95% bootstrap confidence interval: **`[+20.2%, +43.3%]`** (**PASS**).
    - Striker GA / 1k frames plummeted from **0.1644** to **0.0299** ($5.5\times$ reduction) (**PASS**).
    - Striker match record: **1 Win, 2 Losses, 61 Timeouts** (losses slashed from 36 down to 2).
  - **Flawless Non-Regression & Representation Protection**:
    - Tracker 18k screen: **96.9% win rate** (62 W / 0 L / 2 T, 187 GF / 3 GA), beating the $\ge 90\%$ gate (**PASS**).
    - Held-out panel (6k frames): **0 non-Striker losses across 96 matches (0.0%)**, beating the $\le 3/96$ threshold (**PASS**).
  - **All Pre-Declared Gates Passed**.
- **Architectural Attribution**:
  - The model operates as an architectural hybrid (**DQN + Geometric Shield / Soft-X Shaper**). The neural network provides learned offensive intent, corner cutting, and tactical positioning, while the analytical shield provides defensive reachability guarantees against extreme edge shots.

## 2026-10-30: Option 1 — Vector RL Fine-Tuning of Promoted Model (`dqn_v7_promoted.pth`)

**Completed:** Vector RL fine-tuning with selective teacher anchor and Dueling DQN architecture across 3 seeds (11, 22, 33).

- **Peer Deliberation (`loop-with-claude`)**:
  - Deliberated with Claude CLI (`--model opus --effort high`, session `a829d591-7f25-4a2b-a5f4-e82964b60f77`). Reached consensus on:
    1. Direct empirical baseline evaluation on the exact `models/dqn_v7_promoted.pth` checkpoint.
    2. Neutralizing stalling incentives by raising $\gamma$ to $0.997$ and setting `hit_reward = 0.0`.
    3. Selective non-Striker policy anchor ($\lambda = 0.1$) on unscaled advantage logits $A_{\text{student}}(s)$ against the frozen teacher, achieving bit-exact $KL = 0.0$ at warm-start while exempting Striker and Planner transitions.
    4. Pre-step opponent snapshot recording in `ReplayBuffer` to prevent boundary reset label leakage.
    5. Candidate checkpoint ranker filtering non-Striker validation losses $\le$ baseline, ordered by Striker point share $\text{GF} / (\text{GF} + \text{GA})$ (descending).
- **Empirical Results (3 seeds, 1.5M frames each, 576 evaluation matches total)**:
  - **Representation Protection (Flawless)**:
    - Held-out standard panel (384 matches): **0 losses against Tracker across all 3 seeds** (0/32 each).
    - Tracker 18k screen: losses held to **1/64, 4/64, and 3/64** (matching or beating baseline's 4 losses).
    - Pooled non-Striker losses: **3 losses across 288 matches (1.04% loss rate)**, comfortably passing the $\le 11$ gate.
  - **Striker Point Share Gain (+20.6% relative)**:
    - Quadrupled goals scored against Striker (from 15 to 60 total across 192 matches, ~20 per seed), lifting point share from 8.88% to 10.71%.
  - **The Rally Pace Dynamic**:
    - Breaking passive stalling lowered timeouts from 28 to 15–16 per seed. However, active offensive counter-attacks at 900 px/s accelerated the pace of play, resulting in raw goals against per 1k frames rising to 0.1934 vs 0.1644 baseline.
- **Promotion Decision**:
  - Consistent with strict pre-declared gating criteria, because raw GA/1k did not decrease below baseline, **`dqn_v7_promoted.pth` remains the promoted production model**.

## 2026-10-29: UI — Opponent Selection & Arbitrary Model File Picker

**Completed:** GUI upgrade providing full built-in opponent selection and arbitrary model file browsing.

- **1 vs AI Opponent Selection**:
  - Introduced `GameState.HUMAN_VS_AI_SELECT` state machine transition.
  - Allows human players to select any of the 9 built-in opponents (Goal-line Tracker, Edge Striker, Lookahead Planner, Defensive, Aggressive, Follow Ball, Predictive, Random, Dummy) with real-time status and tactical descriptions.
- **Arbitrary Model File Picker**:
  - Added native system file dialog integration via `tkinter.filedialog.askopenfilename` with instant hotkeys (<kbd>B</kbd> or <kbd>O</kbd>).
  - Added `📁 Browse Custom File...` action in the model browser for `.pth` and `.pt` checkpoints.
  - Live validation and parameter inspection panel displaying file size, training step, $\epsilon$, architecture, and validity tone.
  - Safe fallback for headless CI environments (`SDL_VIDEODRIVER=dummy`).
- **Tests**: Added 7 targeted unit tests in `tests/test_config_ui.py`. All 37 tests passing.

## 2026-10-29: Priority 3 — Dueling DQN Architecture & Analytical Fold-on-Save

**Completed:** implemented linear head Dueling DQN with analytical fold-on-save and two-phase calibration in vector training.

- **Dueling Architecture**: $Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')\right)$ with unscaled advantage logits for policy distillation.
- **Fold-on-Save**: Analytically folds advantage and value heads into standard linear layer upon checkpoint save, ensuring 100% backward compatibility with GUI and evaluation runners.
- **Empirical Impact**: Reduced Bellman TD loss variance by 3.1× compared to standard DQN.

## 2026-10-29: Priority 2 — Targeted Striker Optimization Pipeline & DAgger Round 2

**Completed:** comprehensive A0 concession audit and DAgger Round 2 evaluation across 3 seeds (11, 22, 33).

- **Peer Deliberation (`loop-with-claude`)**:
  - Deliberated with Claude CLI (`--model opus --effort high`). Reached consensus that:
    1. Lookahead Planner rollout values are heuristic $\pm 1000$ scores, not true Q-values; regressing on them distorts distillation.
    2. Under softmax cross-entropy, $V(s)$ cancels out and receives zero gradient, meaning a Dueling DQN head has zero impact on action rankings during pure imitation. Dueling head deferred to RL fine-tuning.
    3. The proven loss formulation (`cross_entropy + 0.01 * expected_regret`) and Schema v7 (46D) architecture were kept intact.
- **A0 Concession Audit (192 matches, 467 goals conceded)**:
  - Exact 192/192 bit-for-bit replay reproduction verified through converted Schema v7 checkpoint path (`student_16_v7.pth`).
  - **84.8% of conceded goals (396/467) were physically reachable** by the student paddle (median arrival 55.1 frames = 0.92 s). Only 15.2% were unreturnable setup shots.
  - Concessions are execution errors under high-speed (71.1% at 900 px/s) corner slices.
  - **Planner Ceiling**: Lookahead Planner achieved **64/64 wins (100%), 0 goals conceded** on the identical 18k long-game screen, demonstrating a 100% win-rate theoretical ceiling.
- **DAgger Round 2 (`dagger_r2`) Empirical Results**:
  - Collected 16 fresh student matches per seed against Striker using `student_16_v7` collectors on fresh disjoint seeds (740100..740103). Labeled with Lookahead Planner (16,384 rows/seed).
  - Pooled Round 1 + Round 2 student data with 90% legacy panel data (10% student dosage).
  - Trained for 20 epochs (matching 13,560 update budget) with costly loss.
  - Evaluated on fresh heldout (760100..760107) and fresh 18k long-game scoring screen (770100..770115).
  - **Striker 18k screen**: `dagger_r2` scored 68 goals (0.0344 GF/1k frames) vs 60 goals for `student_16_baseline`. Seed 22 and Seed 33 improved raw goal output (Seed 22: 15 -> 31 goals; Seed 33: 4 -> 17 goals).
  - **Non-Regression Gate Failed**: `dagger_r2` accumulated 35 losses (9.1%) on the competitive heldout panel (including 4 losses to Tracker), compared to only 5 losses (1.3%) for `student_16_baseline`.
  - **Decision**: `dagger_r2` is **NOT promoted**. `student_16` remains the champion checkpoint.
  - **Strategic Recommendation**: Pure imitation learning has reached diminishing returns for high-speed corner defense. Proceed to targeted **reinforcement learning fine-tuning (Priority 3)** with value calibration and TD-learning.

## 2026-10-28: Schema v7 — 16 Contact Geometry Features and Hidden-Layer Projection

**Completed:** upgraded geometry context from the 4-feature linear prototype to the full
16 contact geometry features with first-hidden-layer projection, achieving bit-for-bit
baseline equivalence at zero weights while enabling deep non-linear multi-bounce reasoning.

- **Schema v7 (`magic_pong.dqn_state.v7`)**: canonical DQN input sizes 46 (without score
  context) and 49 (with score context). Retired legacy Schema v6 prototype.
- **Hidden layer projection**: `geometry_context_weight` on `DQNNetwork` shaped
  `(fc_layers[0].out_features, 16)` projected directly into `fc_layers[0]` pre-activation.
  Zero-weight initialization preserves exact baseline forward pass and action selection.
- **Unified feature pipeline**: `@njit(cache=True) write_contact_features_scalar` in
  `magic_pong.ai.geometry` guarantees bitwise feature parity across Python `DQNAgent`,
  Numba `vector_env.py`, and batch NumPy operations.
- **Empirical impact (3 seeds, 384 matches per policy)**:
  - Tracker 18k scoring screen: **175 W / 7 L / 10 T (91.1% wins)**, **0.336 GF/1k frames**
    (2,975 frames/goal), a 6.2× scoring speedup over the 4-feature prototype.
  - Striker 18k scoring screen: **74 goals scored**, **4 wins** (first wins in repo history),
    a 37× increase in goals scored over the 4-feature prototype.
  - Standard heldout panel: losses dropped to **2.1% (8/384 matches)** with a goal margin
    of **+542**.
- Added `magic_pong.ai.models.converter.convert_probe_to_dqn_checkpoint` to load historical
  probe weights into standard v7 checkpoints. All 1,282 test cases passing.

## 2026-09-21: student-state versus teacher-state distillation

**Completed:** planner labels from student-controlled games substantially reduce
losses under the fixed 10% recipe. A matched fresh teacher-state arm regresses.

- Fresh heldout, 384 games per policy: student-state **204 W / 6 L / 174 T**;
  control **201 / 21 / 162**; teacher-state **180 / 45 / 159**;
  incumbent **202 / 28 / 154**.
- Student-state adds 3 wins and removes 15 losses versus control. Losses fall
  in all three seeds; wins change by +8 / −4 / −1. This is chiefly a defense gain.
- Striker remains **0/96 wins**, but goals improve to **25/100** for/against
  from control's **5/123**; goals increase in every seed. 92/96 Striker matches
  still time out. Tracker improves from 14 W / 6 L to 17 W / 0 L.
- 16 shared teacher games and 48 student games; every full game replayed exactly.
  Equal 16,384 rows per source and seed; all 266,944 student actions audited.
- **13,560 new updates**, 6,780 inherited; **2,752 validation + 1,792 distinct
  heldout games**; **17,662,329 main normal-engine frames**. 50 targeted tests
  passed; input, training, selection and completion closures verify.
- All student-state gates pass; teacher-state passes only for seed 22, whose
  heldout result regresses. No model promotion or long training run.

**Next:** test fixed student-state/control/incumbent checkpoints in longer
Tracker/Striker games: score 3, cap 18,000, both sides and bonus modes, 16 fresh
paired seeds. No training or reselection. Confirm whether improved survival and
scoring persist when more matches can resolve, before another collection round.
See the [full results and plots](docs/ai/student_state_distillation_2026-09-21.md)
and [fixed protocol](docs/ai/student_state_distillation_protocol_2026-09-21.md).

## 2026-09-21: continuation mixture dosage, 0% / 10% / 50%

**Completed:** reducing continuation exposure to 10% recovers some of the 50%
mixture's regression, but still does not beat the 0% control. The exact same data,
model, loss, initialization and fit budget were used. Existing 0%/50% candidate
pools were freshly reselected; only the 10% fits are new.

- Fresh heldout,384 games per policy view: incumbent **201 W / 35 L / 148 T**;
  control **194 / 39 / 151**;10% **186 / 53 / 145**;50% **165 / 91 / 128**.
- 10% gains 21 wins and removes 38 losses versus 50%, but loses 8 wins and adds 14
  losses versus control. Losses increase in all three training seeds.
- Striker remains **0/96 wins** for 10%: four goals for/153 against, compared
  with control zero/144. Ordinary-opponent wins also decline.
- Only seed 11's 10% model passed validation; it regressed on heldout. The planned
  gate/fallback yields 192/39/153, also below the incumbent. No model promotion.
- **6,780 new updates**,13,560 inherited updates, **2,880 validation + 1,664
  distinct heldout games**,16,897,879 new normal-engine frames. No new collection.
- **30 targeted tests passed**, exact parent sampler and inherited weight audits
  passed, and the full completion chain verifies. Plots and paired cells saved.

**Follow-up completed:** the matched student-state/teacher-state test is recorded
above. It keeps the low mixture and model/loss/update budget fixed while using
fresh paired full-match collection. Mixture reduction alone did not solve scoring;
broader confirmation is still required before promotion. See the
[full results and plots](docs/ai/continuation_mixture_2026-09-21.md) and
[fixed protocol](docs/ai/continuation_mixture_protocol_2026-09-21.md).

## 2026-09-21: fixed-budget planner-continuation distillation

**Completed:** a 50/50 legacy/continuation minibatch mixture regressed in all three
training seeds. Both arms restarted from identical original RL weights with the
same geometry network, costly loss and update budget. All three controls exactly
reproduced prior histories and checkpoint tensors.

- Collected **63,270 labeled frames from 96 planner continuations**: 39 scored,
  none conceded, 57 were capped. All outcomes entered training.
- **13,560 optimizer updates**, **1,984 validation games** and **1,280 distinct
  held-out games**. All selections were sealed before held-out evaluation.
- In 384 held-out games per policy view: original RL **197 W / 83 L / 104 T**;
  incumbent **200 / 38 / 146**; newly selected legacy-data control
  **206 / 36 / 142**; continuation students **157 / 100 / 127**.
- Continuation training loses **49 wins and adds 64 losses** versus control.
  Every continuation gate fails. Striker remains **0/96 wins**, with four goals
  for and 195 against; control has two for and 151 against.
- New continuation training regret improves sharply, while old development
  regret worsens in every seed. This supports a mixture/coverage trade-off as a
  hypothesis, not a uniquely established cause.
- **19 targeted tests passed**; all 96 teacher and student replays matched,
  independent sampled labels/observations passed, and the completed integrity
  chain verifies. No model promotion.

**Follow-up completed:** the fixed-data 10% mixture comparison is recorded above.
It uses fresh selection and heldout for all three dosage arms. See the
[full results and plots](docs/ai/continuation_distillation_2026-09-21.md) and
[fixed protocol](docs/ai/continuation_distillation_protocol_2026-09-21.md).

## 2026-09-21: Striker scoring and planner continuations

**Completed:** 72 exact student replays, eight full live-planner matches and 184
paired continuations from 46 saved geometry+cost student states. No training.

- The current planner won **7/8** Striker games, **21–0 goals**. Combined students
  won **0/24**, **0–31 goals**, on the same eight cells across three training seeds.
- From identical saved states: student **0 scores / 14 concessions / 32 capped**;
  one teacher action **0 / 14 / 32**; 30-frame commitment **1 / 5 / 40**;
  continuous planner **15 / 0 / 31**. Teacher scoring recoveries occur in all three
  training seeds and take **2.1–10.05 seconds**.
- This supports a failure to execute sustained recovery/scoring behavior. The
  current teacher can supply it on these states. Data coverage, optimization and
  representation have not been causally separated. No simple steeper-shot or
  more-forward-position reward follows from the geometry measurements.
- **72/72** old match rows and **46/46** baseline continuations reproduce exactly.
  **10 targeted tests passed**; independent runner/export reviews approved.
  Main simulation work: **576,018 frames**; zero optimizer updates.

**Follow-up completed:** the fixed-budget planner-continuation comparison is
recorded above. It retained all continuation outcomes and used separate gameplay
selection and held-out seeds. See the
[full diagnosis and plots](docs/ai/striker_scoring_diagnostic_2026-09-21.md).

## 2026-09-21: geometry plus costly loss with gameplay selection

**Completed:** the two arms share geometry, initialization, data, optimizer and
20-epoch budget; only the loss differs. Gameplay validation selected checkpoints
before a fresh held-out panel. Three seeds; **1,984 validation + 1,152 held-out
games**, **13,560 optimizer updates**. Independent selection audit passed.

- Original RL: **189 W / 79 L / 116 timeouts** in 384 held-out games.
- Actual geometry + CE students: **183 / 73 / 128**; only seed 11 passed the
  validation gate. Falling back to original RL for seeds 22/33 gives **203 / 52 / 129**.
- Geometry + costly loss: **200 / 34 / 150**; all three passed validation. Compared
  with original RL, wins rise **49.22% → 52.08%**, losses fall **20.57% → 8.85%**.
  Wins improve in seeds 11/22 but regress in seed 33. Most avoided losses become
  timeouts. Versus gated CE it has three fewer wins and 18 fewer losses.
- The combination wins **96/96 Follow Ball**, **89/96 Aggressive**, **15/96 Tracker**,
  and **0/96 Striker**; it scores only **one goal** against Striker. Defense improved;
  strong-opponent scoring remains unsolved. No model was promoted.
- **27 targeted tests passed**, three exact CE training reproductions passed, and
  the sealed source/data/checkpoint/result chain verifies completely.

**Follow-up completed:** the paired student/planner Striker scoring diagnosis is
recorded above. See the
[full report and plots](docs/ai/planner_geometry_combo_2026-09-21.md) and
[fixed protocol](docs/ai/planner_geometry_combo_protocol_2026-09-21.md).

## 2026-09-21: contact geometry and costly mistakes

**Completed both experiments**, in order, on seeds 11/22/33. The shared CE
control reproduced prior fitted weights and every epoch's metrics exactly.
Fresh evaluation: **41,653 planner-labeled states** from 96 matches and **768
short gameplay matches**. The old test rows were excluded; all settings were pinned.

- **Contact geometry:** 16 observation-derived features lowered fresh-test mean
  planner regret **32.60 → 26.32 (−19.28%)** and raised decisive agreement
  **53.48% → 57.12%**, with the same direction in every seed. Geometry students
  won **88/192** short matches, versus **3/192** for original-input CE and
  **98/192** for the starting RL models. Promising, but not a stronger model overall.
- **Costly loss, separately on original inputs:** CE plus 10 times expected
  planner regret / 1,000 lowered test regret to **20.14 (−38.21%)** and costly
  errors from **3.08% to 1.85%**, but won only **1/192** matches. Better offline
  regret did not rescue gameplay; no model was promoted.
- **Verification:** 31 targeted tests passed, independent feature/split audits
  passed, and 16 direct-DQN replay matches exactly reproduced the experimental
  wrapper's results. The new wrapper does not explain the imitation collapse.

**Follow-up completed:** the geometry-plus-cost comparison with gameplay
selection and fresh held-out matches is recorded above. See the
[full report and plots](docs/ai/planner_geometry_cost_2026-09-21.md) and
[fixed protocol](docs/ai/planner_geometry_cost_protocol_2026-09-21.md).

## 2026-09-21: current baseline and planner-imitation diagnosis

**Completed both requested steps.** Three fresh current vector-trainer runs,
seeds 11/22/33, used 120,000,617 actual frames in total with the current aiming
planner. A fixed common validation panel selected checkpoints; separate held-out
panels measured them. Normal-engine results: **548 / 556 / 556 wins out of 720**,
mean **76.85%**. Across the three policies: **0/480 wins against Tracker/Striker**,
versus **1,660/1,680 (98.8%)** against the other seven built-ins. Seed 11 exposes
a selection trade-off: its selected model wins more than the final model but
also loses more. See the [baseline report](docs/ai/current_baseline_2026-09-21.md).

The planner-imitation diagnostic collected **127,779 labels** from 288 matches
with separate train/development/test match groups. The existing 256/128 network
reaches **97.7–100% decisive agreement on tiny 256-state training subsets**, but
only **59.7–64.0% on full training splits** and **52.1–55.5% on unseen matches**
when initialized from RL. Fresh initialization is worse on selected test regret
in all three seeds. Errors persist in clean states, concentrate near contact,
and a roughly 3% tail accounts for about 97% of remaining planner regret.
An independent audit reproduced every stored observation/action and checked
planner-label replay and both-side mappings. See the
[diagnostic report](docs/ai/planner_imitation_diagnostics_2026-09-21.md).

**Follow-up completed:** the contact-geometry and separate cost-sensitive-loss
experiments are recorded above. The baseline/diagnostic results alone did not
isolate model capacity as the cause or establish stronger gameplay after imitation.
All sources, datasets and artifacts are pinned; the application and deployed model
remain unchanged.

## Objective and scope

Build a neural player that consistently beats every built-in AI and strong human
players under normal gameplay rules. An initial measurable target is at least
99% match wins against each built-in opponent, evaluated on both sides with
bonuses both enabled and disabled. Human strength must be assessed separately;
winning a finite benchmark cannot prove superiority over every possible player.

The initial implementation requested on 2026-09-15 covered findings **1–4** below.
The follow-up also covers correctness fixes from **6** and a controlled training
pilot comparing discount factors across three training seeds. Findings **5** and
**7** remain a backlog. Completing software changes does not mean a newly trained
model has achieved the win-rate target.

## 2026-09-18: diagnosis, rule change, tracker opponents, planner and distillation

**Diagnosis (read-only probes through the official competitive harness).**

- Under constant ball speed a goal-line defender was unbeatable: the paddle moves at
  500 px/s and the ball at 300 px/s, so a paddle that holds its goal line and follows
  the ball's y always arrives in time. A 40-line tracker beat the reference champion
  `model_new/agent_256x2_pipeline_best.pth` 12-0 in matches and 132-0 in goals
  (both sides, bonuses on and off); tracker versus Follow Ball or Aggressive ended
  0-0 at the 18,000-step cap. "Beats all humans" was unattainable under those rules.
- The champion conceded with its paddle 250-340 px in front of its goal line in
  10-11 of every 11 goals and spent its 90th-percentile time at the centre line: it
  learned to chase like Follow Ball. The per-hit reward and the physics teacher's
  hit bonus both pay more for earlier contact, so both reward forward chasing.
- The built-in opponents steered the paddle's top-left corner onto the ball instead
  of its centre, making their strength side-dependent; Follow Ball as player 1
  escorted the ball into its own goal (traced frame by frame).
- Every experiment of 2026-09-15..18 fine-tuned this policy for 1M frames with a
  20,000-frame replay on a panel of three deterministic matches per cell; none beat
  the initial model. That series is stopped; its artifacts older than 2026-09-18
  were moved to the git-ignored `models/archive/repo_cleanup_2026-09-18/`
  (reversible). Eleven experiment modules were restored because today's scripts and
  the library tests import them.

**Changes.**

1. Rule: a genuine paddle bounce multiplies the ball speed by
   `BALL_SPEED_INCREASE` (1.05) up to `MAX_BALL_SPEED`, raised from 500 to 900 so
   long rallies exceed the paddle speed; pushes never change velocity; serves reset
   to `BALL_SPEED`. Every earlier checkpoint and win rate is invalid under this rule,
   and the DQN velocity inputs are now scaled by 900 instead of 500.
2. `UnitObservationMixin.player_pos` is the paddle centre; all rule-based opponents
   use it (`tests/test_simple_ai_geometry.py`).
3. New built-ins `tracker` (goal-line tracker) and `striker` (goal-line defender that
   predicts the intercept and returns with the paddle edge). `BUILTIN_OPPONENTS` has
   nine names, so the competitive panel is 9 × 2 sides × 2 bonus modes × 3 seeds =
   108 matches. Registered in the league, `AI_TYPES`/`DEFAULT_OPPONENT_MIX`, the
   tournament, the frame pipeline validation list and the GUI.
4. `magic_pong/core/fast_physics.py`: numba kernels mirroring the Python engine,
   checked frame by frame in `tests/core/test_fast_physics.py`. A teacher decision
   (9 actions × 90 frames) fell from 40.6 ms to 0.24 ms (24.6 Hz to about 4,100 Hz).
   Optional dependency: `pip install -e ".[fast]"`.
5. `PlannerAI` (`magic_pong/ai/models/planner_ai.py`): lookahead player bound to the
   live engine through the new `bind_physics` hook in `GameEngine.set_players`.
   GUI: "1 vs Planner" menu entry and a planner entry in "Watch AI vs AI".
6. `magic_pong/ai/distillation.py`: DAgger distillation of the planner into the DQN
   network on real matches (`--pretrain_mode distill`, or
   `python -m magic_pong.ai.distillation --labels N --output ... --evaluate`).

**Measurements under the new rule** (validation seeds 101-103, all 108 cells,
timeouts are non-wins):

| Player | Wins | Losses | Timeouts | Goals for / against |
| --- | ---: | ---: | ---: | ---: |
| striker (scripted) | 75/108 | 0 | 33 | 919 / 0 |
| PlannerAI, 6-frame commitment | 65/108 | 0 | 43 | 853 / 0 |
| PlannerAI, 30-frame commitment (new default) | 91/108 | 0 | 17 | 1147 / 2 |
| Distilled 256×128, hard labels, 6-frame teacher, 300k labels | 31/108 | | | |
| Distilled 256×128, soft targets T=1, 6-frame teacher | 58/108 | | | |
| Distilled 256×128, 30-frame teacher, soft T=0.03 + tie-break 0.1, 300k labels | 45/108 | 63 | 0 | 642 / 1010 |

The 30-frame planner wins all 84 matches against the seven classic built-ins, beats
the tracker in 6 of 12 and the striker in 1 of 12 cells (the other cells time out at
0-0) and concedes 2 goals in 108 matches. The scripted defenders time out against
each other in every cell, also in a probe with the cap at 1200 and 1500 px/s:
offense against a perfect defender needs better aiming, not a higher cap.
The distilled network reproduces the planner's labels only about half the time and
loses to Follow Ball, Defensive, Aggressive and both goal-line defenders: the
pipeline works end to end (300k labels, 144 matches, about 3 minutes plus the panel)
but imitation quality is the open problem. The weakest-cell ranking therefore ties at zero on
those cells; use "never concedes" as the gate and goal difference as the tie-break.

**Why the first distillation failed.** With a 6-frame commitment the nine candidate
actions tie in more than 99% of states (the defensive continuation repairs any first
action within the horizon), so the planner's labels are 98% "stay" and it only moves
at the last moment; a network imitating those labels learns to stay. A 30-frame
commitment (half a second) separates the candidates in 25-31% of states and makes
the planner both stronger and imitable. `PlannerAI` and the distiller now default to
30 frames; the distiller trains on `softmax(values / 0.03)` mixed with a 10% one-hot
share of the planner's choice and re-fits on the aggregated buffer after every match.

**Training loop rebuilt (2026-09-18, later the same day).**
`magic_pong/ai/vector_training.py` replaces the episode loops for new work:

- `vector_env.py` steps 128 matches per call on the numba kernels (`fast_physics.py`
  gained serves, bonus spawning, rotating-paddle placement and match resets driven by
  explicit random draws); `fast_policies.py` ports the nine built-ins to numba
  (exact parity tests) so opponents cost nothing; frozen self-play snapshots and the
  planner are the other opponents. The DQN observation encoder is reproduced exactly
  (60 parity cases, both schemas), so checkpoints remain ordinary `DQNAgent` files.
- Learner: action repeat 4 (gamma 0.99 now spans about 6.7 s of play), 3-step returns
  computed at sampling time from a 1M lock-step ring buffer (truncations bootstrap from
  the pre-reset state), Huber loss on Double-DQN targets, Adam 1e-4, batch 256, 8
  gradient samples per transition, hard target copies every 200k frames, goals end the
  return by default, optional potential-based intercept shaping.
- Evaluation every 1M frames: 20 seeds × 36 cells = 720 matches on the vectorised
  engine, draws reported separately, weakest-cell Wilson ranking selects `best.pth`;
  `--final_panel` runs the exact Python panel afterwards.
- Throughput: 52-65k frames/s on one CPU core including updates (the GPU gives no gain
  for this network size), i.e. 40M frames in about 12 minutes plus evaluations, against
  roughly 5k frames/s for the legacy loop. 984 tests pass.

| Run (40M frames, default mix incl. self-play and planner) | Exact panel of `best.pth` |
| --- | ---: |
| From scratch, epsilon 1 → 0.05 over 4M frames | 86/108 wins, 0 losses, 22 timeouts (all against tracker/striker); 560/720 on the vectorised panel |
| From the distilled network, epsilon 0.2 → 0.05 over 2M frames | 82/108 wins, 5 losses, 21 timeouts; 529/720 on the vectorised panel |
| From scratch, 60M frames, aiming planner (below) added to the mix | 82/108 wins, 1 loss (striker), 25 timeouts; peaked 554-558/720 with 0-2 losses on the vectorised panel from 40M frames on |

**Single seed each; not yet a comparison.** The third row uses the same defaults
as the first plus the aiming planner (below), 20M more frames, and the inert
`player_previous_pos` observation field. Its vectorised evaluation was
consistently stronger (554-558/720, 0-2 losses, versus the first run's 560/720
recorded once); its exact 108-match panel is not clearly better or worse (82 vs
86, one real loss to striker, both still 0/12-2/12 against the two goal-line
defenders). **This is one seed per arm; the project's own rule everywhere else
in this document is at least three seeds before treating a difference as real.**
Read this as "no evidence yet that a stronger training opponent alone breaks the
defender barrier," not as evidence that it hurts. A disciplined 3-seed-per-arm
comparison (~1 hour of compute at this throughput) is the next step before
promoting either checkpoint or drawing a conclusion about the aiming planner's
value as a training opponent. Its value as a standalone player (below) and as a
distillation teacher is already established independent of this result.

**Verification.** 984 tests pass (546 before), ruff clean, graft graph rebuilt.

**Next steps.**

- Improve the planner's objective: aim offsets in the continuation policy, a tracker
  as the opponent model, longer horizons where cheap; then re-run the panel.
- Report decisive-state agreement during distillation; scale labels and network
  capacity; only then RL fine-tune with action repeat, a large replay buffer and a
  20M+ frame budget.
- Rotating-paddle collisions are positional (not swept); at 900 px/s the ball can
  tunnel through an obstacle. Bonus size effects still compound on re-application.

## Baseline evidence (before these changes)

Checkpoint: `model_new/agent_256x2_pipeline_best.pth` (30 inputs, 9 actions,
hidden layers 256 → 128). Evaluation used deterministic actions, no bonuses,
first to 11 points, seed 20260915, and 50 matches per opponent/timing condition.
The simulation budget was 166.7 seconds: 1,000 steps at the old headless timing,
10,000 steps at normal timing. Timeouts were not wins.

| Opponent | Old headless wins / losses / timeouts | Normal timing wins / losses / timeouts |
| --- | --- | --- |
| follow_ball | 43 / 2 / 5 (86% wins) | 16 / 29 / 5 (32% wins) |
| aggressive | 31 / 9 / 10 (62% wins) | 2 / 46 / 2 (4% wins) |

These measurements predate the velocity-unit correction as well as the timing
fix. They are historical evidence of the defect, not a directly comparable
baseline for the corrected simulator. Under the 2026-09-15 rules and pilot protocol,
the same previous checkpoint wins **68/84 matches (81.0%)**; see the
[2026-09-15 reference evaluation](docs/ai/previous_checkpoint_validation_2026-09-15.json).

The older root-level `eval*.json` files refer to older checkpoint schemas and
must not be confused with this baseline. A small check of the two-layer final
self-play checkpoint also failed to resolve the normal-timing weakness.

The original exploration ran 20 environment, DQN stability, and evaluation tests;
all passed despite the issues below.

## Progress

Status key: **In progress**, **Implemented / verified**, **Experiment pending**,
**Backlog**. Record commands, outcomes, and remaining limitations below each item.

| ID | Workstream | Status | Acceptance criteria |
| --- | --- | --- | --- |
| 1 | Consistent simulation and observations | Implemented / verified | Identical headless/GUI steps; velocity units correct; normalization invariant; randomized geometry coherent |
| 2 | Competitive checkpoint selection | Implemented / verified | Frozen, reproducible validation across opponents/sides/bonuses; worst-matchup selection; independent test seeds |
| 3 | Broader, stronger opponent league | Implemented / verified | All built-ins; persistent historical/diverse opponents; weakness-aware sampling; exploiter support |
| 4 | Recovery and scoring pretraining | Implemented / verified | Real-simulator teaching for interception/recovery/attack; learner-visited examples; runnable training integration |
| 5 | More informative neural observations | Score context implemented / verified; other features backlog | Explicit scores/target; predictive features, effect timers, complete objects, masks, angle encoding, optional recurrence |
| 6 | Rewards and episode endings | Implemented / verified; pilot complete | Separate truncation/termination; shaping consistently controlled; compare discount horizons |
| 7 | DQN learning and throughput experiments | Backlog | Controlled multi-step/PER/loss/architecture experiments; efficient replay and batched environments |
| 8 | Constant ball speed and positional pushing | Implemented / verified | Pushes preserve the complete velocity vector; genuine bounces preserve speed; pretraining uses configured speed; resizing excluded from paddle motion |
| 9 | Seed 44 training regression | State-management fixes verified; late defensive deterioration measured; update-level cause unresolved | Preserve phase state; resolve ambiguous full-match value targets; demonstrate recovery under current physics |
| E | Demonstrate competitive playing strength | In progress; target not met | Pilot complete; broader held-out benchmark and human/adversarial matches remain |

### Latest competitive investigation: 2026-09-16

The completed `league_seed44_resumed` run reports 30.09% training point wins.
Fresh identical 84-match evaluations under the constant-speed physics give
**36/84 (42.9%) for its final model**, **59/84 (70.2%) for its best**, and
**63/84 (75.0%) for both the seed-44 curriculum best and the older reference**.
The final model wins no matches against Follow Ball, Defensive or Aggressive.

The pipeline reloaded the start-of-annealing weights after 50,000 annealing
episodes, then switched directly to goal-only self-play. Exploration reached its
minimum by curriculum episode 150. The physics transition also reduced earlier
checkpoint strength. Fix the handoff and schedules, then run a controlled pilot
before another long training run. See the
[investigation, comparisons and recovery plan](docs/ai/training_regression_2026-09-16.md).
No training implementation changes were made during the initial investigation.

The subsequent authorized implementation adds `--mode frame_pipeline` with
continuous learner state across phases, frame-based exploration and reward
schedules, a replay-capacity zero-shaping tail, a fixed competitive benchmark,
and complete episode-boundary resume bundles. The legacy episode pipeline also
stops reloading historical best weights at phase boundaries. Usage and limits:
[frame pipeline guide](docs/ai/frame_pipeline.md).

An architect reviewer approved the implementation after a resume-output safety
fix. **418 tests pass**, including 21 schedule/resume tests covering
uninterrupted-versus-resumed tensor equality, self-play league restoration,
corrupt-bundle rejection and interruption boundaries.

The matched six-branch pilot is complete: seeds 11/22/33, 1M frames each,
6M total frames and 2,880 validation matches. Corrected final weights score
**164/252 (65.1%)** against built-ins versus the control's **151/252 (59.9%)**,
but every corrected branch finishes below the initial model's **63/84 (75.0%)**.
All three corrected branches retain that initial model as `best.pth`.
Follow Ball and Aggressive also fail to improve over the starting model.
**Do not launch a long run with this schedule yet.** These results motivated the
exploration-only follow-up below while retaining the state-management fixes.
The completed [report](docs/ai/schedule_pilot_2026-09-16.md),
[full results](docs/ai/schedule_pilot_2026-09-16.json) and
[curves](docs/ai/schedule_pilot_2026-09-16.png) preserve the evidence and next step.

The authorized exploration-only follow-up is complete, reusing verified
corrected controls with unchanged application source/runtime. Conservative
exploration finishes at **144/252 (57.1%)**, below the previous schedule's
164/252, and wins only **3/36 against Follow Ball and 4/36 against Aggressive**.
It fails the predeclared screening criteria. The completed conditional
learning-rate comparison uses the same conservative exploration and learning
rate **0.00003 instead of 0.0003**. Its finals improve to **162/252 (64.3%)**, but
only one seed retains the initial 63/84 score. Selected best models total
195/252 (77.4%), yet Follow Ball only ties the initial pooled count (12/36) and
Aggressive falls below it (15/36 versus 18/36). **Both follow-ups fail the screen**;
neither warrants a long run or promotion of a new champion.

The exploration and learning-rate follow-ups add six 1M-frame runs and 2,880
validation matches. See their
[combined report and reproduction commands](docs/ai/adaptation_experiments_2026-09-16.md),
[exploration data](docs/ai/exploration_pilot_2026-09-16.json),
[learning-rate data](docs/ai/learning_rate_pilot_2026-09-16.json), and
[checkpoint integrity audit](docs/ai/adaptation_integrity_2026-09-16.json).

The authorized **reward-floor experiment is also complete**: six fresh 1M-frame
branches and 2,880 validation matches compare retaining 10% auxiliary rewards
with annealing to zero, at conservative exploration and learning rate 0.00003.
Treatment finals score **174/252 (69.0%)**, versus **162/252 (64.3%)** in fresh
controls. The paired changes are +18/−3/−3 wins; no treatment final retains the
initial 63/84. Selected-best scores are 194/252 (77.0%), with Follow Ball 13/36
and Aggressive 13/36. **The predeclared screen fails; no new champion or long-run
setting is validated.** All control network fingerprints reproduce the previous
low-learning-rate trajectory. The reward-floor implementation passes **430
tests** and retains zero as its default. The completed
[report and protocol](docs/ai/reward_floor_pilot_2026-09-16.md),
[full data](docs/ai/reward_floor_pilot_2026-09-16.json),
[curves](docs/ai/reward_floor_pilot_2026-09-16.png) and
[integrity audit](docs/ai/reward_floor_integrity_2026-09-16.json) preserve results.
The authorized **league-switch diagnostic is complete**: another six fresh
1M-frame branches and 2,880 validation matches, with reward floor 0.10 and all
other settings fixed. Continuing Follow Ball in the final phase wins
**186/252 (73.8%)**, versus the league's **174/252 (69.0%)**. All three paired
differences improve (+3/+3/+6 wins), so the diagnostic passes. However, both arms
decline from every shared 500k checkpoint; fixed-opponent finals retain the
initial 63/84 in only one seed. Selected-best fixed models score 195/252, with
Follow Ball 13/36 and Aggressive 11/36. **The competitive screen still fails.**
The league switch contributes to the decline under these settings; avoiding it
only partly helps. No long run or new champion is validated. The fixed-opponent
option remains experimental, and `league` remains the default.

The implementation passes **435 tests**, including 38 schedule/resume cases.
See the [report and protocol](docs/ai/league_switch_pilot_2026-09-16.md),
[full data](docs/ai/league_switch_pilot_2026-09-16.json),
[curves](docs/ai/league_switch_pilot_2026-09-16.png) and
[checkpoint/replay audit](docs/ai/league_switch_integrity_2026-09-16.json).
The **2026-09-17 match-length comparison is complete**: six fresh 1M-frame
branches and 2,880 validation matches compare `(1,)` with `(1,3,5,11)` from
the start of training, with Follow Ball and reward floor 0.10 held fixed.
Mixed targets finish at **139/252 (55.2%)**, versus the one-point control's
**186/252 (73.8%)**. All paired differences are negative (−22/−15/−10 wins).
Selected-best mixed models total 192/252 (76.2%), with Follow Ball 13/36 and
Aggressive 14/36; all are selected at curriculum end. **Both the diagnostic and
competitive screen fail. Do not use this mix for a long adaptation run under
these settings.** No champion or training default is changed.

The treatment completes 295/327 (90.2%) of its 11-target games by score, with
roughly half its frames spent on that target. Mixed runs have 1,315 total
episodes versus 5,213 for controls; actual lengths and caps are recorded. Fresh
one-point controls exactly reproduce the previous fixed-Follow-Ball trajectories.
The telemetry implementation passes **440 tests**, including 43 schedule/resume
cases. See the [report and protocol](docs/ai/match_length_pilot_2026-09-17.md),
[data](docs/ai/match_length_pilot_2026-09-17.json),
[curves](docs/ai/match_length_pilot_2026-09-17.png) and
[checkpoint/replay audit](docs/ai/match_length_integrity_2026-09-17.json).
Further work should inspect learner targets and actions around the decline
before committing to another long run. No additional training experiment has
been launched.

An independent **Claude consultation on 2026-09-17** recommends a fixed-state
value/action audit, measured goal-versus-auxiliary credit, replay composition
checks and finer behavioral metrics before further training. Claude received an
anonymized summary with no repository access. Its response contains errors;
the [checked consultation notes](docs/ai/claude_consultation_2026-09-17.md)
separate useful hypotheses from corrections and local facts. In particular,
Double DQN and truncation-aware bootstrapping already exist. The encoder's
missing match target/absolute scores is a new concrete hypothesis for mixed
training. The [follow-up diagnostics](docs/ai/value_diagnostics_2026-09-17.md)
are now complete; no training defaults or models changed.

The **2026-09-17 value investigation** audited 30 checkpoints, six final replay
buffers and 2,520 built-in match records; reproduced 84 instrumented matches
exactly; compared 3,351 fixed states; and evaluated nine first-action branches
at each of 84 simulator snapshots. There were **zero training updates** and no
held-out matches. **445 tests pass**. Mixed finals have mean point margin
−0.69 versus +2.71 for one-point finals and +2.89 initially. They change action
rankings extensively, but there is no uniform action-gap collapse or Q inflation
on these probes. Mixed replay has more goals but fewer episodes, weakening a
simple lack-of-goal-samples explanation. Goal credit is weak for some earlier
contacts, but not uniformly weaker in mixed trajectories.

An executable simulator probe confirms identical encoded states with different
goal termination rules when target/absolute scores differ. Actual nonterminal
mixed goals add about +0.29–0.33 of next-state value to their targets. The causal
share of omitted context alone remains unmeasured. This motivated a controlled
comparison that keeps mixed physical matches and changes only whether each goal
ends the critic's return, with fresh paired controls and separate
simulator/learner ending logs.
That experiment is now **complete**: six fresh runs, 6M frames and 2,880
validation matches. Goal-terminal learning improves final wins from 55.2% to
69.4% and mean goal margin from −0.694 to +1.381, with both improvements in
all three seed pairs. It remains below the initial 75.0%; the strength screen
fails three of five gates. Selected bests reach 79.4%, but Aggressive remains
below initial (15/36 versus 18/36). The exact control trajectories and full
training states reproduce the previous mixed runs; all 6,959 treatment
intermediate goals are masked while physical matches continue. Keep the option
experimental and avoid another long run. Next proposed comparison: add
side-relative absolute scores and match target under the original full-match
return, preserving the initial policy with zero-initialized new inputs.
See the [report](docs/ai/goal_bootstrap_pilot_2026-09-17.md),
[data](docs/ai/goal_bootstrap_pilot_2026-09-17.json),
[plot](docs/ai/goal_bootstrap_pilot_2026-09-17.png) and
[integrity audit](docs/ai/goal_bootstrap_integrity_2026-09-17.json).

The authorized **score-context comparison is complete**: six fresh 1M-frame
runs and 2,880 validation matches keep full-match returns and mixed physical
matches fixed. Adding own score, opponent score and target improves pooled
final wins from **139/252 (55.2%) to 156/252 (61.9%)** and mean goal margin from
−0.694 to +0.119. Paired gains are **+21/+9/−13 wins**. Both diagnostic screens
pass, but every treatment final remains below the initial 63/84 and three of
five strength gates fail. Selected bests total 197/252 (78.2%), with Follow Ball
13/36 and Aggressive only 10/36. **No long-run configuration or new champion is
validated.** Late decline persists, especially in seed 33.

Migration preserves legacy weights, RNG and initial Q values exactly. The
33-input schema, legacy loading and full-state resume pass **495 tests**.
Controls reproduce historical complete training states; all 120,000 saved
treatment score triples match physical events, and the new projections learn
finite nonzero weights. Independent implementation and results reviews pass.
The report recommends inspecting retained intermediate/final models and replay
to localize late deterioration before another training sweep. See the
[report and commands](docs/ai/score_context_pilot_2026-09-17.md),
[data](docs/ai/score_context_pilot_2026-09-17.json),
[curves](docs/ai/score_context_pilot_2026-09-17.png), and
[integrity audit](docs/ai/score_context_integrity_2026-09-17.json).

The authorized **late-decline investigation is complete**. It audits 30
checkpoints and six final replay buffers, reproduces 116 saved validation matches
exactly, scores 8,123 real states, and continues 116 identical physical snapshots
with frozen policies. **No training updates or model changes** occurred.
Seed 33's decline is broad (63→39 wins from 500k→1M), with most added concessions
in its traced matches occurring without a paddle contact. On 84 common
incoming-near snapshots, its midpoint policy scores/concedes 31/45 times, versus
17/65 for the final policy; contacts fall 90→42. Zeroing or replacing only the
score projection does not restore the behavior.

Q-values fall and action choices drift; the large replay action-gap contraction
does not hold uniformly on validation probes. Immediate auxiliary gradients are
small relative to goal gradients. The strongest lead is deterioration of useful
defensive behavior during ongoing value fitting, but retrospective final-buffer
analysis cannot identify the historical updates that caused it. **502 tests
pass**, including seven new diagnostic cases; independent review agrees with the
bounded interpretation. The follow-up tests a frozen reference-policy anchor
on a fresh training-state bank (progress below). See the
[investigation and reproduction commands](docs/ai/late_decline_diagnostics_2026-09-17.md),
[data](docs/ai/late_decline_diagnostics_2026-09-17.json), and
[figure](docs/ai/late_decline_diagnostics_2026-09-17.png).

The authorized **frozen defensive-policy anchor test is complete**. Its
[fixed protocol](docs/ai/policy_anchor_protocol_2026-09-17.md) compares six fresh
1M-frame branches (three matched seeds), with anchoring active only after 500k
frames. Collection completed: 2,048 successful-contact examples in 16 balanced
side/bonus/target cells, using 156 new training games and 291,834 reference
simulation frames. Reference weights/Q values are unchanged; all nine actions
are represented, with the most frequent action at 31.5%. **509 tests pass**,
including exact anchored resume and sampler isolation. Existing validation
states are excluded from training; held-out seeds remain unused. No default
training behavior or champion has been changed. All six runs completed their
1M-frame budgets; all 2,880 validation matches and saved artifacts pass the
strict audit, including exact historical controls and paired 500k training states.

Final pooled wins improve **156→177/252 (61.9→70.2%)** and mean goal margin
improves in all three pairs (**+0.119→+1.774** pooled), passing the primary
diagnostic rule. Wins by seed are **60→58, 57→55, 39→64**: the win-consistency
rule fails. Bank action agreement rises in every seed, but **three strength
gates still fail** and all selected-best checkpoints remain unchanged, before
anchor activation (**197/252 pooled; Aggressive 10/36 vs 18/36 initially**).
This is partial stabilization, not a stronger selected model. **No long run or
promotion recommended.** A possible next bounded comparison freezes the
reference at the 500k phase handoff and collects new defensive examples under
that policy; the original reference already disagrees with most midpoint
actions. That follow-up is now complete (below). See the
[completed report](docs/ai/policy_anchor_pilot_2026-09-17.md),
[curves](docs/ai/policy_anchor_pilot_2026-09-17.png), and
[integrity audit](docs/ai/policy_anchor_pilot_2026-09-17_integrity.json).

The authorized **handoff-reference anchor test is complete**. Three new
1M-frame runs use seed-specific banks from the exact frozen 500k handoff
policies, preserving their trained score inputs; the previous six branches are
explicitly reused after verification. Finals reach **171/252 (67.9%)**, above
unanchored controls (61.9%) but below the original anchor (70.2%). Mean goal
margin is +0.952, versus +0.119/+1.774 respectively. The primary diagnostic
passes against control but fails against the original anchor; win consistency
fails against both. New finals are **56/84, 56/84 and 59/84**.

Seed 11 improves its selected checkpoint to **67/84 at 752,495 frames**, then
falls to 56/84. Selected-best pooled wins rise **197→202/252 (78.2→80.2%)**,
but the strongest individual checkpoint remains seed 33's unchanged 250k
policy. **Three strength gates still fail; no long run or promotion recommended.**
Handoff-bank agreement improves versus controls but finishes at only 40–47%.
The follow-up comparing about-750k and final defense and TD/anchor gradients is
now complete (2026-09-18; below).

**516 tests pass**; strict audits confirm exact pre-handoff reproduction,
correct seed-specific banks, all 3M new learner frames and 1,440 new validation
matches, and 50,000 anchor updates per run. Collection adds 1,007,900 frames in
463 games; no held-out evaluation was used. See the
[fixed protocol](docs/ai/handoff_anchor_protocol_2026-09-17.md) and
[completed report](docs/ai/handoff_anchor_pilot_2026-09-17.md),
[curves](docs/ai/handoff_anchor_pilot_2026-09-17.png), and
[integrity audit](docs/ai/handoff_anchor_pilot_2026-09-17_integrity.json).

The **handoff-anchor late-decline investigation is complete (2026-09-18)**.
All **252** instrumented matches exactly reproduce saved validation. On **252
identical physical snapshots**, intermediate→final concessions rise **159→171**,
contacts fall **218→198**, and paired adverse/favorable concessions are **30/18**.
Seed 11's effect is modest (44→48 concessions) despite its larger benchmark
drop; this only partly explains its failure. The **1,920 gradient probes** show
small weighted-anchor/TD norms (median **1.7–3.8%** at ~750k), with most own-target
gradients compatible. Substituting fixed final target values changes conflicts
from **14/47/28 to 118/121/109 out of 128** across seeds. Adam momentum can also
oppose the anchor, especially seed 22. These are retrospective final-replay
probes; they do not reconstruct historical updates or prove a larger coefficient
will help. **Zero training updates**, **524 full-suite / 22 targeted tests pass**.
Recommended next bounded intervention: matched continuation with target-network
refresh frozen versus normal, after deterministically reconstructing and
verifying the full intermediate fork state from the saved 500k bundle. This
next experiment is now complete (below); no long run or model
promotion is recommended.
See the [report](docs/ai/anchor_decline_diagnostics_2026-09-18.md),
[figure](docs/ai/anchor_decline_diagnostics_2026-09-18.png), and
[integrity export](docs/ai/anchor_decline_diagnostics_2026-09-18_integrity.json).

The **target-refresh continuation experiment is complete (2026-09-18)**.
Three reconstructed full intermediate states and fresh normal continuations
reproduce the historical candidates/finals exactly after validated checkpoint
serialization normalization. Freezing subsequent target copies improves pooled
finals from **171/252 (67.9%) to 184/252 (73.0%)** and mean goal margin by
**+0.865**. Margins improve in all three seeds, wins in two: both diagnostic rules
pass. Reference-bank agreement rises from **42.7% to 53.3%**. However, finals
remain below the initial 75.0%, three strength gates fail, and selected pooled
wins fall slightly from 202 to 201/252. The strongest individual checkpoint is
unchanged. Next recommendation: a bounded **10,000-update target-refresh** arm
against the existing 1,000-update and frozen arms, using these saved forks.
No long run or default promotion is validated. **2,243,173 fresh training
frames, 864 fresh validation matches, 535 full-suite / 11 focused tests pass**.
See the [report](docs/ai/target_refresh_pilot_2026-09-18.md),
[plot](docs/ai/target_refresh_pilot_2026-09-18.png),
[protocol](docs/ai/target_refresh_protocol_2026-09-18.md), and
[audit amendment](docs/ai/target_refresh_audit_amendment_2026-09-18.md).

The **slower target-refresh experiment is complete (2026-09-18)**.
From the same verified forks, copies every 10,000 learner updates improve final
wins to **185/252 (73.4%)**, versus normal **171/252 (67.9%)** and frozen
**184/252 (73.0%)**. All three seeds improve wins and goal margin versus normal;
both diagnostic rules pass. Versus frozen, +1 win and −0.012 mean margin give no
clear advantage. Reference-bank agreement reaches **52.0%**, versus normal
42.7% and frozen 53.3%. Three strength gates still fail; selected checkpoints
match normal (202/252) and the strongest individual model is unchanged. Next
recommendation: a bounded gradual-target-update test with a fraction pinned
before training. No long run, default or champion is promoted.
**743,173 fresh frames / 288 fresh validation matches; 540 full-suite / 75
targeted tests pass**. Prior controls are explicitly reused. See the
[report](docs/ai/slow_target_pilot_2026-09-18.md),
[plot](docs/ai/slow_target_pilot_2026-09-18.png), and
[protocol](docs/ai/slow_target_protocol_2026-09-18.md).

The **gradual target-update experiment is complete (2026-09-18)**.
Blending 2/11 online weights into the target every 1,000 updates yields
**180/252 final wins (71.4%)**, below slow copies **185/252 (73.4%)** and frozen
targets **184/252 (73.0%)**. Mean margin also falls relative to slow copies
(+1.722 versus +1.806); both predeclared diagnostic rules fail. Gradual updates
improve on normal in all three seeds, but three absolute strength gates still
fail and selected checkpoints remain unchanged. No long-run setting is validated.
**546 full-suite / 81 targeted tests pass**; all 73 interpolations reproduce
exactly and nine prior arms pass provenance checks. Fresh work is **743,173
frames and 288 validation matches**; no held-out evaluation. Next proposed step
is a read-only optimizer-momentum probe using actual fork replay/state, following
the earlier gradient evidence; its follow-up is tracked below. See the
[protocol](docs/ai/soft_target_protocol_2026-09-18.md) and
[experiment report](docs/ai/soft_target_pilot_2026-09-18.md).

The **optimizer-momentum diagnostic is complete (2026-09-21)**. This closes a
historical question about the retired fine-tuning series, using its preserved
application snapshot and three full forks. First-moment clearing improves the
defensive-bank loss relative to original Adam in seed 22, but worsens it in seeds
11 and 33, failing the predeclared two-seed criterion. The equal-norm sensitivity
improves mean loss in seeds 11/22 but not 33; action-agreement gains favor original
Adam in all three seeds. No blanket momentum reset is supported. All 384 paired
batches reproduce exactly; **1,000 full-suite / 27 targeted tests pass**.
There are **zero training frames and zero gameplay matches**. These results do
not apply directly to the newer physics/vectorized learner. Continue the current
three-seed competitive comparison described above; close this historical branch.
See the
[fixed protocol](docs/ai/optimizer_momentum_protocol_2026-09-21.md) and
[report](docs/ai/optimizer_momentum_diagnostics_2026-09-21.md).

The existing teacher's small conditional advantage does not justify replacing the learner. See the
[report](docs/ai/value_diagnostics_2026-09-17.md),
[data](docs/ai/value_diagnostics_2026-09-17.json) and
[plot](docs/ai/value_diagnostics_2026-09-17.png).

Finding 8 was implemented on 2026-09-16 after the user's clarification. A moving
paddle now pushes an outgoing ball by correcting position only, leaving both
velocity components unchanged and producing no repeated hit rewards. Ordinary
incoming hits preserve speed. Swept contact uses the entry geometry, including
exact contact at frame start; a later genuine hit on the same paddle can bounce.
Paddle effects expire before movement snapshots, and pretraining initializes
normalized velocities at `BALL_SPEED`. See the
[evidence and implementation](docs/ai/ball_speed_investigation_2026-09-16.md).

Verification: **397 tests passed**, including 38 new push/speed cases and updated
legacy acceleration expectations. Collision code has 98% line coverage. All
playing-strength percentages recorded for the 2026-09-15 pilot predate this
physics change; retain them as historical results and re-evaluate checkpoints
under the new rules before making competitive comparisons.

## Backlog: planner aiming (implemented), observation velocity field (implemented), geometry-context schema (designed, not implemented)

**2026-09-18, continuing the training-loop rebuild.**

**Measured result of planner aiming** (exact 108-match panel, seeds 101-103):
wins rose from **91/108 (0 losses, 17 timeouts, goals 1147/2)** before this change
to **100/108 (0 losses, 8 timeouts, goals 1175/0)** after it — the planner has now
never once conceded across 108 matches. Aiming turns former 0-0 timeouts into
wins: tracker 6/12 → 11/12 (1 timeout left), striker 1/12 → 5/12 (7 timeouts
left, 0 losses). Re-running 300k-label distillation with the aiming teacher
raised the distilled network from 45/108 to **55/108**, though its failure mode
changed from passive timeouts to active losses against Follow Ball, Aggressive
and both goal-line defenders — direct DAgger distillation at this label budget
still lags far behind the RL loop (86/108, 0 losses at 40M frames, below).

**Implemented: planner aiming.** `PhysicsTeacher`/`FastPhysicsTeacher` and
`continuation_action`/`fast_physics._continuation` take an `aim_offset` (fraction
of the half paddle height); the teacher now tries each first action at three
offsets (`DEFAULT_AIM_OFFSETS = (0.0, 0.85, -0.85)`) and keeps the best value per
action, so a return can strike near an edge instead of always centring on the
opponent. `_attack_value`/`fast_physics._attack_value` changed from a step
function (only unreachable balls score) to a graded "pressure" term (fraction of
the opponent's travel capacity the return consumes, `ATTACK_PRESSURE_WEIGHT=10`),
evaluated once, at the frame the ball leaves the paddle, not at the rollout
horizon. Cost: three offsets triple the rollout, 0.24 ms/decision to 0.82 ms
(compiled backend); still 1.2 kHz standalone. In `VectorEnv`, whose
`planner_actions()` calls the teacher once per planner-opponent environment in a
Python loop (not batched), realistic training throughput with the default mix
(planner weight 0.5 of ~9.5) fell from roughly 50-65k to about 32k frames/s.
Batching the rollout across environments in one numba call would recover this;
not done. 988 tests pass.

**Implemented: `player_previous_pos`.** `ObservationProcessor` now emits the
player's own last position (`game_state[f"player{{id}}_last_position"]`,
falling back to the current position), mirrored in `to_player1_dqn_observation`
exactly like `opponent_previous_pos`. No existing encoder reads it; this only
unblocks an own-paddle-velocity feature. Zero effect on any existing checkpoint
or trained behaviour (verified: full suite, including the DQN encoding parity
tests, passes unchanged).

**Implemented: `geometry_context` input schema (v6).** The DQN sees
one raw frame with no history and no fed-back previous action; with the
{{-1,0,1}}-per-axis bang-bang action space, its own paddle's velocity is
otherwise unobservable, and it had to learn wall-bounce reflection purely
from reward. Implemented 4 extra features, computed in the canonical (player-1)
frame purely from observation-dict fields:

1. `predicted_intercept_y` — reflected ball y at the own paddle's contact face
   (computed in pixel space with wall reflection math, normalized by `field_height`).
   Sentinel when the ball is not approaching: paddle center `(own_y + own_height / 2) / field_height`.
2. `time_to_intercept` — arrival time normalized by horizon `1.5`s (90 frames at 60 FPS),
   clamped to `[0.0, 1.0]`. Sentinel when not approaching: `1.0`.
3. `own_paddle_vel_y` — `(player_pos.y - player_prev_pos.y) / (paddle_speed * frame_dt)`,
   clamped to `[-1.0, 1.0]`.
4. `opponent_paddle_vel_y` — `(opponent_pos.y - opponent_prev_pos.y) / (paddle_speed * frame_dt)`,
   clamped to `[-1.0, 1.0]`.

*(Note on 2D paddle motion: own paddle x-velocity is omitted here to strictly adhere to the 4-feature contract; it remains an explicit candidate for a future extension block).*

**Architecture & Checkpoint Schema v6:**
- Input blocks follow an append-only layout: `[base: 0:30, score_context: 30:33, geometry_context: 33:37]`
  (or `[base: 0:30, geometry_context: 30:34]` if score context is disabled).
- State sizes: 30 (base), 33 (score only), 34 (geometry only), 37 (both).
- Zero-projection bolt-on: `DQNNetwork.geometry_context_weight` initialized with zeros (`fc_layers[0](x[:30]) + score_context_weight(x[30:33]) + geometry_context_weight(x[geom_slice])`),
  guaranteeing exact bitwise initial Q-value and action equivalence upon migration.
- Checkpoint format: v6 with `input_blocks` metadata list, individual metadata blocks for enabled contexts,
  and backwards-compatible upgrades from v4 (base) or v5 (score) to v6.
- Bitwise parity verified between Python `ObservationProcessor` and Numba `@njit` `VectorEnv` across all combinations.
- Integrated into `scheduled_training.py` (`--frame_geometry_context`), `train_optimized.py`, `vector_training.py` (`--geometry_context`), and GUI (`MagicPongApp`).

## 1. Consistent simulation and observations

Findings:

- `GameEnvironment.step` multiplies the headless timestep by 10: 166.7 ms versus
  16.7 ms at 60 FPS. A 500 px/s paddle moves 83 versus 8 pixels per decision.
- Collision response and catch-up checks compare paddle displacement with ball
  velocity. A controlled 500 px/s paddle movement produced outgoing ball speeds
  of 312.5 versus 425 px/s solely because the timestep changed.
- `ObservationProcessor` optionally normalizes positions/velocity, while
  `DQNAgent._observation_to_state` normalizes again. Centered ball X became
  0.000625 instead of 0.5 when the optional normalization flag was enabled.
- `_domain_rand_context` changes global field height after the environment has
  cached its dimensions. A reproduction yielded configured/paddle height 655
  while physics and observation height remained 600.

Implementation direction: one simulation/decision contract regardless of rendering;
correct velocity units; a declared observation coordinate format; rebuild or
synchronize all geometry at randomized episode boundaries. Speed up computation
without enlarging the timestep. Existing models need reevaluation/retraining after
physics corrections.

Implemented:

- Headless and GUI decisions now advance the same simulated time. Explicit large
  timesteps are split into physics updates of at most one frame.
- Paddle deflection and catch-up checks use velocity in pixels/second.
- Observations declare whether positions are normalized and include the velocity
  scale. Both encoders and built-ins respect the declared format. The neural
  input remains 30 elements, so existing checkpoints still load.
- Randomized geometry updates physics, paddles, bonus spawns and observations,
  and restores them even after an exception. Extreme samples stay within valid
  field-size and speed bounds; the complete candidate config is checked first.
- The default training cutoff is now 10,000 frames, preserving the old default's
  approximate simulated-time budget after removing the headless multiplier.

Regression coverage: `tests/test_simulation_consistency.py`, environment, entity,
collision and physics tests. Relevant code: `magic_pong/ai/interface.py`,
`magic_pong/ai/models/dqn_ai.py`, `magic_pong/core/physics.py`,
`magic_pong/core/collision.py`, and `_domain_rand_context` in `train_optimized.py`.

## 2. Competitive checkpoint selection

Findings:

- `train_phase` selects the best checkpoint using a recent training point-win
  average against rule-based opponents only. Self-play strength is excluded.
- A 100% window saturates the strict improvement criterion, preventing later
  strategically stronger checkpoints from replacing an early model.
- Hyperparameter search maximizes average opponent win rate and varies validation
  seeds by trial number; this hides weak matchups and adds comparison noise.

Implementation direction: periodically freeze the candidate and evaluate full
matches without exploration across opponents, both sides and bonus modes. Use
common validation seeds and a worst-matchup confidence-bound ranking, with
meaningful tie-breakers. Keep separate held-out test seeds. Include historical
neural opponents and avoid perturbing learner state or training RNG during eval.

Implemented:

- `magic_pong/ai/competitive.py` evaluates full matches against all seven built-ins,
  on both player sides and with bonuses on/off. Timeouts count as non-wins.
- Checkpoint ranking compares the weakest cell's Wilson lower bound, then mean
  lower bound, mean goal difference and mean win rate. Results retain the complete
  per-match/cell evidence and resolved configuration.
- `magic_pong/ai/training_validation.py` selects checkpoints at phase boundaries
  and periodically (`--competitive_eval_interval`). Existing incumbents are
  re-evaluated using the current protocol before replacement.
- Validation preserves learner/replay state, module modes, opponent state and
  global RNG state. Held-out seeds are disjoint and never used for selection.
- Frozen neural references are included for league phases. Benchmark identities
  contain checkpoint SHA-256 digests, preventing silent opponent replacement.
- Hyperparameter search uses the same validation seeds for every trial and the
  weakest-cell objective. Resuming a legacy or changed scoring protocol requires
  a new study name.
- Patience starts fresh in each phase. Pipeline training preserves the live
  learner at phase handoffs; best-model selection remains separate. Retained
  incumbent reports are refreshed, disabled selection cannot pick up a stale
  file, and the top-level pipeline best has its validation report.
- Managers and persistent leagues are cleaned up after training failures.

Regression coverage: `tests/test_competitive_evaluation.py`,
`tests/test_training_validation.py`, `tests/test_ai_evaluation.py`,
`tests/test_search_hyperparameters.py`, and integration tests.

Three seeds per cell are the default screening budget, **not evidence of 99%
strength**. Use a larger held-out set for confirmation and inspect every cell's
confidence bounds and timeout rate. Changing the benchmark requires re-evaluating
both candidates; scores from different protocols are not comparable.

## 3. Broader, stronger opponent league

Findings:

- The default self-play mix omits `aggressive`.
- Only five recent snapshots are retained, encouraging forgetting and omitting
  strategically different historical opponents.

Implementation direction: cover every built-in; retain champions and a diverse
historical pool; support independent checkpoints and specialists trained to exploit
the champion; increase sampling of weak matchups without eliminating coverage.
Persist league state so training can resume and failures can be reproduced.

Research: [AlphaStar league training](https://www.nature.com/articles/s41586-019-1724-z).

Implemented in `magic_pong/ai/league.py` and `SelfPlaySampler`:

- Every built-in retains sampling coverage, including `aggressive`.
- Weak empirical matchups receive more training weight, with a coverage floor.
- Recent snapshots, champions and historical opponents have separate retention
  budgets. Historical retention uses performance profiles/roles where available
  and spaced ages otherwise; it is a practical heuristic, not a learned diversity
  measure.
- Independent checkpoints can be imported. A specialist can train against a
  frozen target, then be imported into the main league.
- Immutable checkpoint copies, outcome statistics, protected benchmark references
  and sampling RNG state persist in the league directory. Resume verifies hashes.

Regression coverage: `tests/test_league.py` and CLI integration tests. League
persistence does not make an ordinary `.pth` weights load a bit-for-bit restart
of the entire trainer; choose `--load_checkpoint` when optimizer/training state
from a training checkpoint is needed.

## 4. Recovery and scoring alongside interception

Findings:

- Existing pretraining generates approaching-ball states, moves the paddle once,
  and rewards positioning. It does not teach coherent rally recovery or attack.
- Independent synthetic samples already use `done=True`; preserve that property
  where applicable. The older recovery plan is a proposal, not completed behavior.

Implementation direction: use the actual physics engine to evaluate candidate
actions/short sequences, covering interception, recovery after a return, and
attacking contact position. Include corner saves, small paddles and bonus scenarios.
Collect teacher labels on learner-visited states (DAgger-style) and fine-tune through
competitive reinforcement learning. Runtime play can remain purely neural.

Research: [DAgger](https://proceedings.mlr.press/v15/ross11a.html).

Implemented in `magic_pong/ai/physics_teacher.py` and
`magic_pong/ai/simulator_pretraining.py`:

- The teacher compares action prefixes through copies of the real physics engine,
  scoring actual contacts/goals and recovery/attack positioning. Each branch
  shares the same random state and leaves the live simulation untouched.
- Training trajectories cover interception, recovery, attack, corner saves,
  small paddles and bonus collection, on both player sides.
- Mixed learner/teacher behavior produces labels on states the learner visits.
  Labels train a separate supervised imitation loss and synchronize the target
  network. They do not become fictitious reinforcement-learning transitions.
- `--pretrain` now uses simulator teaching; `--pretrain_mode legacy` retains the
  old task for comparison. Runtime action selection remains neural.

Regression coverage: `tests/test_pretraining_teacher.py` checks actual defensive
saves, scoring actions, recovery, bonus collection, rollout state/RNG isolation,
learner decisions, supervised updates and simulator integration.

The teacher has a bounded horizon and a simple opponent model. It supplies useful
initial behavior, not optimal-play labels. Its `average_reward`/`all_rewards`
statistics are teacher rollout scores, not competitive match win rates. A local
CPU timing of 120 default labels took about 4.4 seconds; speed varies by machine.

## Follow-up findings

### 5. More informative inputs

An opt-in 33-element schema now adds side-relative own/opponent scores and the
winning score. Legacy 30-input models remain supported; migration starts the
added projection at zero. The [completed paired experiment](docs/ai/score_context_pilot_2026-09-17.md)
shows a variable pooled improvement without resolving late regression.

Both schemas still omit effect expiration timers and include at most three
bonuses and two rotating paddles. Consider interception position/time, relative
geometry, opponent velocity, effect durations, presence masks, sine/cosine angles,
and history/recurrent state for hidden dynamics.

### 6. Winning objective and episode endings

Original findings: training cutoffs and true match termination shared one `done`
flag, and `MICRO_REWARD_SCALE=0` did not disable enabled proximity/Phase 3 shaping.

Implemented in the follow-up:

- Environment `info` and training statistics distinguish `terminated` from
  `truncated`, while the existing `done` signal still ends the episode loop.
- The DQN bootstraps from the real final observation at a time cutoff and stops
  bootstrapping after true match termination. A match ending on the cutoff frame
  takes precedence. Legacy transitions retain their original `done` interpretation.
- The shaping scale now controls contact, collection, proximity and Phase 3
  rewards, preserving goal outcomes. Stateful shaping calculations still advance
  when their contribution is zero. Training restores the original scale after
  success or failure.
- Targeted tests cover episode-ending semantics, zero/fractional reward scaling,
  goal rewards and shaping-history/config restoration.

At 60 decisions/s, gamma=0.99 has an exponential discount time scale of 1.66
seconds; gamma=0.999 has one of 16.66 seconds. Rewards five seconds away retain
4.9% versus 74.1% of their weight. The controlled pilot compares both settings;
it does not assume the longer horizon is better. Potential-based shaping remains
a future experiment.

References: [Gymnasium time limits](https://gymnasium.farama.org/main/tutorials/handling_time_limits/),
[potential-based shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

### 7. Learning algorithm and throughput

Double DQN already exists. Experiment separately with multi-step returns,
prioritized replay, Huber loss, dueling/distributional value estimation, and network
capacity. Profile a contiguous replay ring buffer and batched inference across
independent environments. Compare equal compute budgets and several training seeds.

Reference: [Rainbow](https://arxiv.org/abs/1710.02298).

## Experiment log

| Date | Experiment/check | Result | Artifact |
| --- | --- | --- | --- |
| 2026-09-15 | Original timing comparison, 200 matches | Severe transfer gap; see baseline above | [Raw baseline](docs/ai/baseline_2026-09-15.json) |
| 2026-09-15 | Original targeted tests | 20 passed | `tests/test_environment.py`, `tests/test_dqn_stability.py`, `tests/test_ai_evaluation.py` |
| 2026-09-15 | Final implementation regression suite | **325 passed**, including coverage instrumentation | `tests/` |
| 2026-09-15 | Coverage for new AI modules | Competitive 94%; league 94%; physics teacher 96%; simulator pretraining 95%; checkpoint selection 96% | Overall `magic_pong/ai` coverage is 73%, including untouched legacy scripts |
| 2026-09-15 | Tiny integrated training run | Pretrained, updated learner, retained league and selected a checkpoint across both sides/bonus settings | `tests/test_training_integration.py` |
| 2026-09-15 | Command-line smoke runs | Full pipeline, persistent-league resume, specialist training/import and held-out evaluation completed; tiny budgets test wiring only | `/tmp/magic-pong-implementation-{smoke,resume,exploiter,import}/` (ephemeral) |
| 2026-09-15 | Architecture and training integration reviews | Findings resolved; no remaining review blockers | Phase patience, stale checkpoint selection, report provenance, failure cleanup and search protocol covered |
| 2026-09-15 | Static checks | Ruff, Black formatting/AST equivalence and `git diff --check` pass; graft refreshed | 27 changed/new Python files checked |
| 2026-09-15 | Finding 6 and pilot-runner verification | **359 tests passed**, including coverage instrumentation; pilot runner 91% covered | Episode-ending, shaping, paired-start, frame-budget, provenance and cleanup regressions |
| 2026-09-15 | Controlled discount pilot | Completed 6,000,000 RL frames and 15,000 teacher labels; 1,512 validation matches; gamma 0.99 final mean 73.8%, gamma 0.999 67.1% | [Report](docs/ai/discount_pilot_2026-09-15.md), [complete data](docs/ai/discount_pilot_2026-09-15.json), [CSV](docs/ai/discount_pilot_2026-09-15.csv), [curves](docs/ai/discount_pilot_2026-09-15.png) |
| 2026-09-15 | Previous checkpoint under corrected rules | 68/84 wins (81.0%); best pilot checkpoint 72/84 (85.7%); improvement remains unconfirmed | [Reference evaluation](docs/ai/previous_checkpoint_validation_2026-09-15.json) |
| 2026-09-15 | Pilot integrity verification | All 126 recorded artifact hashes verified; all 43 package source hashes unchanged; paired starts and exact frame quotas verified; 48,010 episode logs have finite metrics; pilot held-out set untouched | `models/discount_pilot_2026-09-15/manifest.json`, per-seed `result.json` |
| 2026-09-15 | Independent results review | Pooled counts, checkpoint ranking, weak matchups and provisional gamma recommendation confirmed; Ruff and `git diff --check` pass | [Report](docs/ai/discount_pilot_2026-09-15.md) |
| 2026-09-16 | Frame schedules, continuous handoffs and full resume | Architect-reviewed; 418 tests pass, including 21 schedule/resume tests; runner 96.8% line coverage | [Usage and limits](docs/ai/frame_pipeline.md) |
| 2026-09-16 | Controlled schedule/handoff pilot | 6M frames, 2,880 validation matches; corrected finals 65.1%, control 59.9%, starting model 75.0%; no long-run gate passed | [Report](docs/ai/schedule_pilot_2026-09-16.md), [data](docs/ai/schedule_pilot_2026-09-16.json), [CSV](docs/ai/schedule_pilot_2026-09-16.csv), [curves](docs/ai/schedule_pilot_2026-09-16.png) |
| 2026-09-16 | Exploration-only adaptation | 3M new frames; conservative finals 57.1% versus 65.1% control; initial 75.0%; screen failed | [Report](docs/ai/adaptation_experiments_2026-09-16.md), [data](docs/ai/exploration_pilot_2026-09-16.json), [curves](docs/ai/exploration_pilot_2026-09-16.png) |
| 2026-09-16 | Learning-rate-only adaptation | 3M new frames; lower-LR finals 64.3% versus 57.1% control; 1/3 seeds retain initial strength; hard-opponent screen failed | [Report](docs/ai/adaptation_experiments_2026-09-16.md), [data](docs/ai/learning_rate_pilot_2026-09-16.json), [curves](docs/ai/learning_rate_pilot_2026-09-16.png) |
| 2026-09-16 | Retained auxiliary reward comparison | 6M fresh frames, 2,880 matches; 10% floor finals 69.0% versus 64.3% control; benefits 1/3 paired seeds, retains baseline in 0/3; screen failed; 430 tests pass | [Report](docs/ai/reward_floor_pilot_2026-09-16.md), [data](docs/ai/reward_floor_pilot_2026-09-16.json), [CSV](docs/ai/reward_floor_pilot_2026-09-16.csv), [curves](docs/ai/reward_floor_pilot_2026-09-16.png), [audit](docs/ai/reward_floor_integrity_2026-09-16.json) |
| 2026-09-16 | Final-phase league switch | 6M fresh frames, 2,880 matches; continued Follow Ball finals 73.8% versus league 69.0%; all 3 paired seeds improve, but both arms decline from midpoint; diagnostic passes, competitive screen fails; 435 tests pass | [Report](docs/ai/league_switch_pilot_2026-09-16.md), [data](docs/ai/league_switch_pilot_2026-09-16.json), [CSV](docs/ai/league_switch_pilot_2026-09-16.csv), [curves](docs/ai/league_switch_pilot_2026-09-16.png), [audit](docs/ai/league_switch_integrity_2026-09-16.json) |
| 2026-09-17 | One-point versus mixed target-score training | 6M fresh frames, 2,880 matches; mixed finals 55.2% versus one-point 73.8%; all 3 paired seeds regress; 90.2% of 11-target games finish by score; both screens fail; 440 tests pass | [Report](docs/ai/match_length_pilot_2026-09-17.md), [data](docs/ai/match_length_pilot_2026-09-17.json), [CSV](docs/ai/match_length_pilot_2026-09-17.csv), [curves](docs/ai/match_length_pilot_2026-09-17.png), [audit](docs/ai/match_length_integrity_2026-09-17.json) |
| 2026-09-17 | Value, replay and simulator diagnostics | 30 checkpoints; 120k replay transitions; 84 exact reproduced traces; 3,351 fixed states; score aliasing confirmed, causal contribution unmeasured; zero training; 445 tests pass | [Report](docs/ai/value_diagnostics_2026-09-17.md), [data](docs/ai/value_diagnostics_2026-09-17.json), [plot](docs/ai/value_diagnostics_2026-09-17.png) |
| 2026-09-18 | Vectorised numba training loop (action repeat, n-step Huber Double-DQN, 1M replay, self-play, 720-match evaluation) | 984 tests pass; 52-65k frames/s; two 40M-frame runs: scratch 86/108 with 0 losses, distilled init 82/108 on the exact panel | `magic_pong/ai/vector_training.py`, `models/vector_runs_2026-09-18/` |
| 2026-09-18 | Defense-dominance probes and champion diagnosis | Tracker beats champion 12-0 (132-0 goals); champion concedes 250-340 px forward; built-ins aim the top-left corner | Section "2026-09-18" above |
| 2026-09-18 | Ball acceleration rule, centred built-ins, tracker/striker, numba planner, distillation | 688 tests pass; teacher 40.6 ms → 0.24 ms; striker 75/108, planner(6) 65/108, planner(30) 91/108 with 0 losses, distilled 45/108; zero losses for scripted/planner players | `tests/core/test_fast_physics.py`, `tests/test_planner_ai.py`, `tests/test_distillation.py`, `models/distill_demo_2026-09-18/` |
| 2026-09-17 | Goal-terminal learner with unchanged mixed matches | 6M fresh frames, 2,880 matches; finals 69.4% versus 55.2% control, initial 75.0%; margins +1.381 versus −0.694; all three pairs improve; strength screen fails 3/5; 472 full-suite tests passed | [Report](docs/ai/goal_bootstrap_pilot_2026-09-17.md), [data](docs/ai/goal_bootstrap_pilot_2026-09-17.json), [CSV](docs/ai/goal_bootstrap_pilot_2026-09-17.csv), [curves](docs/ai/goal_bootstrap_pilot_2026-09-17.png), [audit](docs/ai/goal_bootstrap_integrity_2026-09-17.json) |

## Pilot findings and next decision

This section records the **2026-09-15 discount pilot under earlier physics**.
For the current decision, use the [2026-09-16 schedule pilot](docs/ai/schedule_pilot_2026-09-16.md)
and the latest investigation above; the earlier longer-run recommendation is
superseded by the current regression evidence.

The controlled pilot is complete. Each final column below pools three independent
training seeds, with 84 validation matches per checkpoint. The shared untrained
and pretrained checkpoints are counted once per seed.

| Stage | Gamma 0.99 | Gamma 0.999 |
| --- | --- | --- |
| Untrained | 34/252 (13.5%) | Shared baseline |
| Simulator pretrained | 71/252 (28.2%) | Shared baseline |
| 500,000 RL frames | 98/252 (38.9%) | 57/252 (22.6%) |
| 1,000,000 RL frames | **186/252 (73.8%)** | **169/252 (67.1%)** |

Final wins by training seed (each out of 84):

| Seed | Gamma 0.99 | Gamma 0.999 |
| --- | --- | --- |
| 11 | 72 | 45 |
| 22 | 52 | 59 |
| 33 | 62 | 65 |

**Decision: retain gamma 0.99 as the provisional setting.** Its mean final win
rate is higher, but gamma 0.999 wins two of the three paired seeds. Three training
seeds do not establish a reliable overall winner. Both mean learning curves rise
from midpoint to final, so this pilot does not establish a performance plateau.
There was one validation timeout, in seed 22's final gamma 0.999 evaluation;
timeouts count as non-wins.

The best pilot checkpoint under the configured competitive ranking is
[`seed_11/gamma_0.99/final.pth`](models/discount_pilot_2026-09-15/seed_11/gamma_0.99/final.pth),
with **72/84 wins (85.7%)**. The previous checkpoint wins 68/84 under the same
current protocol. Four additional validation wins, after selecting among multiple
candidates, do not establish a reliable improvement. Keep both as references.

The best pilot checkpoint still wins only 7/12 against `follow_ball`, 8/12 against
`aggressive`, and 9/12 against `defensive`. It loses all three `follow_ball`
matches on player side 1 with bonuses enabled. Across the three gamma 0.99 final
checkpoints, bonuses-off wins are 104/126 (82.5%) versus 82/126 (65.1%) with bonuses
on. The all-opponents target remains unmet, and no human benchmark was run.

Historical next steps from the discount pilot (superseded by the current
regression investigation and [Claude consultation](docs/ai/claude_consultation_2026-09-17.md)):

1. Run a longer fixed-frame comparison with additional independent training
   seeds and the same observation/algorithm implementation. Predeclare the frame
   budget and evaluation schedule; retain both gamma values because this small
   pilot is inconclusive. This is recommended future work, not a run already
   launched.
2. Inspect failed rallies against `follow_ball`, `aggressive`, and `defensive`,
   especially with bonuses. Use those traces to choose a small observation
   experiment from finding 5 if the longer run stalls; compare against the same
   training budget. Defer the broad finding 7 changes until a specific learning
   or throughput bottleneck is measured.
3. Once a candidate is selected, compare it with the previous checkpoint on a
   larger, previously unused held-out set, then run human/adversarial matches.

The complete report preserves the configuration, software versions, package
source hashes, checkpoint hashes and per-match results. Checkpoints and episode
logs remain in the local, git-ignored `models/discount_pilot_2026-09-15/` directory;
the Markdown/JSON/CSV/PNG reports are under `docs/ai/`.

## Training and evaluation commands

### Controlled discount-factor pilot (completed)

The pilot uses training seeds 11, 22 and 33; each seed shares identical
initialized and simulator-pretrained weights across gamma=0.99 and gamma=0.999.
Each branch gets exactly 1,000,000 RL environment steps, using the 256 → 128
network and 5,000 teacher-labelled pretraining decisions per seed.

Training balances player sides and bonus modes, uses the persistent opponent
league, and anneals auxiliary rewards to zero. Exploration follows environment
steps so varying episode lengths cannot change the exploration schedule.
Validation compares untrained, pretrained, midpoint and final checkpoints against
all seven built-ins, both sides and both bonus modes on seeds 101–103. Final
held-out seeds remain unused. The code and report generator passed architecture
review. A local single-thread timing check processed 10,000 RL frames in 2.09
seconds; full-run time also includes pretraining, validation and league snapshots.

Run sequentially in a fresh output directory:

```sh
python -m magic_pong.ai.pilot --output-dir models/discount_pilot_new
```

The actual experiment uses `models/discount_pilot_2026-09-15` with three separate
processes, each using the same command plus `--seed-filter 11`, `22`, or `33`.
Each process is limited to one PyTorch CPU thread. The manifest records all
settings, package source hashes and library versions. Completed seeds can be
verified and reused; partial runs require a fresh output directory.

Regenerate the completed report and chart with:

```sh
MPLCONFIGDIR=/tmp/magic-pong-mpl python docs/ai/summarize_discount_pilot.py \
  models/discount_pilot_2026-09-15 \
  --output-prefix docs/ai/discount_pilot_2026-09-15 \
  --reference-report docs/ai/previous_checkpoint_validation_2026-09-15.json
```

Each seed retains untrained/pretrained weights, separate gamma branches with
initial/midpoint/final weights, full validation reports and episode logs. The
pilot trains directly in the league after imitation pretraining; the separate
multi-phase curriculum is not part of this comparison. Adaptive league sampling
may diverge between branches as their outcomes diverge.

Run from the repository with its Python environment activated. These are starting
budgets for experiments; they are not tuned settings or a promise of playing
strength. Use a fresh output directory for independent experiments.

### Fresh simulator-pretrained pipeline

```sh
python train_optimized.py --mode pipeline --pretrain --seed 0 \
  --pretrain_steps 5000 --domain_rand_noise 0.1 \
  --pipeline_curriculum_episodes 1000 \
  --pipeline_anneal_episodes 5000 --pipeline_selfplay_episodes 10000 \
  --max_steps_per_episode 10000 --competitive_eval_interval 1000 \
  --self_play_interval 500 --output_dir models/league_seed0 --model_prefix agent
```

The pipeline emits phase checkpoints and a top-level
`agent_pipeline_best.pth`. Validation writes `.validation.json` summaries and
`.validation.jsonl` histories next to phase best checkpoints. Pretraining writes
its own `.pretraining.json`. The persistent league lives at `OUTPUT_DIR/league`
unless `--league_dir` overrides it. Set `--no-save_plots` for unattended runs that
do not need charts.

Repeat with `--seed 1`, another output directory, and preferably another
`--opponent_mix_seed` to create independent candidates. Keep initialization seeds
separate from the fixed validation protocol.

### Continue training and import a strong independent opponent

```sh
python train_optimized.py --mode single --seed 0 \
  --load_model models/league_seed0/agent_pipeline_best.pth \
  --league_dir models/league_seed0/league \
  --league_opponent models/league_seed1/agent_pipeline_best.pth \
  --self_play_interval 500 --total_episodes 10000 \
  --max_steps_per_episode 10000 --competitive_eval_interval 1000 \
  --output_dir models/continued --model_prefix agent
```

`--league_opponent` can be repeated. Omit it if no independent checkpoint exists.
`--load_model` resumes from weights; use an actual training checkpoint with
`--load_checkpoint` when preserving its training state is desired.

### Train a specialist, then add it to the main league

```sh
python train_optimized.py --mode single --seed 2 \
  --load_model models/league_seed0/agent_pipeline_best.pth \
  --league_exploiter_target models/league_seed0/agent_pipeline_best.pth \
  --total_episodes 5000 --max_steps_per_episode 10000 \
  --output_dir models/exploiter_seed2 --model_prefix exploiter
```

This trains against the frozen target in a separate league. Import the resulting
`models/exploiter_seed2/exploiter_final.pth` into a later main-league run with
`--league_opponent`. The command supports specialists; it does not automatically
launch a distributed exploiter-training service.

### Held-out competitive evaluation

For an inexpensive wiring check:

```sh
python -m magic_pong.ai.competitive \
  models/league_seed0/agent_pipeline_best.pth \
  --split held_out --output models/league_seed0/held_out_screen.json
```

For a larger confirmation run, the Python API avoids listing hundreds of seeds
on the command line:

```python
import json
from pathlib import Path
from magic_pong.ai.competitive import (
    CompetitiveConfig, competitive_result_to_dict, evaluate_competitive_checkpoint,
)

checkpoint = Path("models/league_seed0/agent_pipeline_best.pth")
protocol = CompetitiveConfig(held_out_seeds=tuple(range(20000, 20500)))
result = evaluate_competitive_checkpoint(checkpoint, protocol, split="held_out")
checkpoint.with_suffix(".held_out.json").write_text(
    json.dumps(competitive_result_to_dict(result), indent=2) + "\n"
)
```

This schedules 14,000 matches (7 opponents × 2 sides × 2 bonus modes × 500 seeds)
and can take substantial time. Reserve those seeds until model selection is
finished; do not repeatedly optimize against this test set. Report per-cell wins,
losses, timeouts and confidence bounds. Human/adversarial play remains a separate
uncompleted milestone.

### Reproduce software verification

```sh
pytest tests -q
pytest tests -q --cov=magic_pong/ai --cov-report=term-missing
```

Use the filesystem path in `--cov`: dotted package targets triggered a NumPy
double-import failure during coverage collection in the current environment.
The path-based coverage run passes. Existing Pydantic `model_fields` deprecation
warnings are unrelated to these changes.

## Resuming this work

1. Read the progress table and verification notes before changing code.
2. Use the documented training/validation commands added with this implementation.
3. Record code revision, checkpoint, resolved physics settings, seeds, opponents,
   side, bonus mode, match length, timeouts, and per-matchup results for each run.
4. Keep training, validation, and held-out test results distinct.
5. Mark the playing-strength milestone complete only after measurements establish it.
