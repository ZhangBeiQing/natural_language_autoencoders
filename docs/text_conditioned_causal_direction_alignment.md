# Text-Conditioned Causal Direction Alignment

This document describes a concrete research plan for combining NLA-style
text-to-activation reconstruction with RDO-style causal direction learning.
The goal is to train a model that maps natural-language concepts into clean,
causally effective activation directions for a specific target model.

## Objective

Given a trained AR model from NLA:

```text
AR(text, layer_id) -> activation vector
```

continue training it into a concept controller:

```text
goal text + layer_id + mode -> intervention direction
```

The generated direction should be useful for activation addition or ablation:

```text
add(v)     -> model behavior moves toward the goal
ablate(v)  -> model behavior moves away from the concept
```

Unlike original NLA, the success criterion is not just activation
reconstruction. The generated vector must cause the target model to change
behavior in the intended direction while preserving unrelated capabilities.

## Core Idea

Use NLA as pretraining and RDO as alignment:

```text
NLA pretraining:
  text <-> activation
  learns the target model's hidden-space coordinate system

RDO / behavior alignment:
  text -> causally effective direction
  learns which directions actually control model behavior
```

This is analogous to language-model pretraining followed by preference
alignment. NLA gives the AR a broad semantic map of the target model's
activation space; the alignment stage teaches it which parts of that space are
causal, clean, and usable for intervention.

## Model Setup

Use three components:

```text
1. Frozen target LM
   The model whose behavior will be controlled.

2. Trainable AR/controller
   Initialized from the NLA AR checkpoint.

3. Optional teacher directions
   RDO, DIM, SAE, or contrastive directions for known concepts.
```

Recommended first version:

```text
target LM: frozen Qwen2.5/Qwen3-style model
controller: existing NLA AR + LoRA or trainable vector head
output: one direction for one selected layer
```

After the single-layer version works, extend to:

```text
AR(goal_text, layer_id) -> v_layer
```

and later:

```text
AR(goal_text) -> {layer_i: v_i}
```

Do not assume that the same vector is valid across all layers. Different layers
have different representation spaces.

## Training Data

Each training example should describe a target behavior, a prompt where that
behavior matters, and a way to evaluate whether intervention helped.

Recommended JSONL schema:

```json
{
  "concept_id": "admit_uncertainty",
  "goal": "Admit uncertainty when the answer is not known.",
  "layer": 20,
  "mode": "add",
  "prompt": "...",
  "good_response": "...",
  "bad_response": "...",
  "retain_prompt": "...",
  "retain_response": "...",
  "teacher_vector_path": "optional/path/to/rdo_vector.pt"
}
```

Fields:

- `goal`: natural-language control request.
- `prompt`: prompt where the concept should affect behavior.
- `good_response`: target behavior after intervention.
- `bad_response`: undesired behavior.
- `retain_prompt`: unrelated normal prompt used to check side effects.
- `teacher_vector_path`: optional RDO/DIM direction for distillation.

Start with concepts that are easy to evaluate:

```text
admit uncertainty
avoid hallucination
be concise
avoid sycophancy
be scientifically skeptical
correct mistakes
avoid overconfidence
remain calm
use step-by-step reasoning
```

For each concept, create many paraphrases of the goal text. Example:

```text
Admit uncertainty when unsure.
Do not pretend to know facts you cannot verify.
Say when evidence is insufficient.
Avoid fabricating citations or confident guesses.
```

These paraphrases should map to similar intervention directions.

## Intervention Operators

Normalize the generated direction:

```text
u = v / (||v|| + eps)
```

Addition:

```text
h_layer = h_layer + scale * u
```

Ablation:

```text
h_layer = h_layer - projection(h_layer, u)
projection(h, u) = (h dot u) * u
```

The first experiment should use addition at one layer. Once stable, test
ablation and multi-layer interventions.

## Alignment Loss

The controller is trained through the frozen target LM. The target LM weights
do not update, but gradients flow from the target LM logits through the
intervention hook into the controller output vector.

### Preference Loss

Use a pairwise objective. The generated vector should make the target model
prefer the good response over the bad response.

```text
score_int =
  logP_intervened(good_response | prompt)
  - logP_intervened(bad_response | prompt)

score_base =
  logP_base(good_response | prompt)
  - logP_base(bad_response | prompt)

delta = score_int - score_base

L_pref = -log sigmoid(beta * delta)
```

Using `delta` is important: it trains the controller to produce the behavioral
change caused by the vector, not merely to rely on what the base model already
prefers.

### Retain Loss

For unrelated prompts, intervention should not damage normal behavior.

```text
L_retain = KL(
  P_base(. | retain_prompt)
  ||
  P_intervened(. | retain_prompt)
)
```

For efficiency, compute this over selected target tokens or the final N tokens,
not necessarily over the entire sequence.

### Teacher Direction Loss

If a concept has an RDO/DIM teacher vector:

```text
L_vec = 1 - cosine(v, v_teacher)
```

This should be a helper loss, not the whole objective. The main objective is
behavioral effect.

### Regularization

Norm control:

```text
L_norm = (||v|| - target_norm)^2
```

Optional orthogonality against unrelated known directions:

```text
L_orth = sum_i cosine(v, unrelated_direction_i)^2
```

Total loss:

```text
L =
  lambda_pref   * L_pref
  + lambda_retain * L_retain
  + lambda_vec    * L_vec
  + lambda_norm   * L_norm
  + lambda_orth   * L_orth
```

Initial weights:

```text
lambda_pref   = 1.0
lambda_retain = 1.0
lambda_vec    = 0.2
lambda_norm   = 0.01
lambda_orth   = 0.05
beta          = 0.1 to 1.0
```

## Training Loop

One training step:

```text
1. Encode goal text with the AR/controller.
2. Extract generated vector v for the requested layer.
3. Normalize v into direction u.
4. Run frozen target LM with intervention.
5. Compute logP for good and bad responses.
6. Run frozen target LM without intervention for baseline scores.
7. Compute retain KL on unrelated prompts.
8. Add optional teacher-vector and regularization losses.
9. Backpropagate only into AR/controller parameters.
```

Pseudocode:

```python
target_lm.requires_grad_(False)
controller.train()

for batch in loader:
    v = controller(batch["goal"], batch["layer"])
    u = normalize(v)

    good_int = target_lm_logp(
        prompt=batch["prompt"],
        response=batch["good_response"],
        hook=add_direction(batch["layer"], u, scale),
    )
    bad_int = target_lm_logp(
        prompt=batch["prompt"],
        response=batch["bad_response"],
        hook=add_direction(batch["layer"], u, scale),
    )

    with torch.no_grad():
        good_base = target_lm_logp(batch["prompt"], batch["good_response"])
        bad_base = target_lm_logp(batch["prompt"], batch["bad_response"])

    delta = (good_int - bad_int) - (good_base - bad_base)
    loss_pref = -F.logsigmoid(beta * delta)

    with torch.no_grad():
        retain_base = target_lm_logits(batch["retain_prompt"])

    retain_int = target_lm_logits(
        batch["retain_prompt"],
        hook=add_direction(batch["layer"], u, scale),
    )
    loss_retain = kl_divergence(retain_base, retain_int)

    loss = loss_pref + lambda_retain * loss_retain

    if "teacher_vector" in batch:
        loss += lambda_vec * (1 - cosine(v, batch["teacher_vector"]))

    loss += lambda_norm * norm_penalty(v)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Minimal Experiment

Run the first experiment on one concept only:

```text
concept: admit uncertainty
target model: Qwen model used by NLA
layer: current NLA extraction layer
mode: add
trainable params: LoRA or final vector head only
loss: L_pref + L_retain + L_norm
```

Evaluation:

```text
1. Does addition increase good-vs-bad logprob margin?
2. Does generated text become more calibrated on held-out prompts?
3. Does retain KL stay small?
4. Does the vector transfer across paraphrases of the same goal?
5. Does it fail on unrelated concepts?
```

Only after this passes should the experiment expand to multiple concepts.

## Multi-Concept Expansion

After the single-concept run:

```text
1. Train on 10-20 concepts with clear good/bad pairs.
2. Hold out several concepts entirely.
3. Test whether the controller can generate useful directions for held-out
   concepts from natural-language descriptions alone.
4. Add teacher-vector distillation for concepts that have RDO/DIM directions.
5. Add layer conditioning.
6. Add multi-layer output.
```

Important generalization tests:

```text
seen concept + new prompt
seen concept + paraphrased goal
held-out concept + normal prompt
composed goal, e.g. "be concise and admit uncertainty"
negative control, e.g. random vector or unrelated goal vector
```

## Success Criteria

A generated direction is considered useful only if it passes all three checks:

```text
1. Behavioral effect:
   intervention improves target behavior on held-out prompts.

2. Retention:
   unrelated tasks remain close to the base model.

3. Specificity:
   the direction affects the intended concept more than unrelated concepts.
```

Do not judge success by vector similarity alone. Cosine similarity to an RDO
direction is useful diagnostics, but the final metric is causal behavior under
intervention.

## Relationship to Existing NLA Training

Original NLA:

```text
activation -> explanation -> reconstructed activation
```

This project:

```text
goal text -> intervention direction -> changed model behavior
```

The AR checkpoint from NLA is valuable because it already knows how natural
language maps into the target model's activation space. The alignment stage
changes what "correct" means:

```text
before: vector matches original activation
after: vector causes intended behavior with low side effects
```

That is the main conceptual bridge between NLA and RDO.

## Practical Notes

- Keep the target LM frozen during the first version.
- Train only AR LoRA or a small vector head at first.
- Start with one layer and one intervention mode.
- Use behavior loss as the primary signal.
- Use RDO/DIM directions as teacher data when available.
- Track retain KL on every experiment.
- Save every generated vector with its goal text, layer, scale, eval scores,
  and training checkpoint.
- Prefer held-out concepts and paraphrases for measuring true generalization.

The long-term target is a controller that can convert natural-language goals
into target-model-specific activation directions with measurable causal effect,
not merely directions that look semantically plausible.
