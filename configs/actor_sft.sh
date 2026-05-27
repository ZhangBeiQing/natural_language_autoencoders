#!/bin/bash
# Actor SFT — teach the actor to explain an injected activation.
# Debug signal: if injection breaks, actor outputs Chinese (literal ㊗ in context).

: "${AV_SFT_PARQUET:?set AV_SFT_PARQUET to the Stage 3a parquet path}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-${BASE_MODEL:-}}"
: "${INSTRUCT_MODEL:?set INSTRUCT_MODEL to the HF checkpoint (must have nla_meta.yaml)}"
: "${SAVE_DIR:?set SAVE_DIR for output}"
: "${INJ_SCALE:?set INJ_SCALE — injection hyperparameter (e.g. 1.0, 30.0, raw, sqrt_d_model)}"

${PYTHON:-python} train.py \
    --train-backend "${TRAIN_BACKEND:-fsdp}" \
    --custom-actor-cls-path "${ACTOR_CLS:-nla.train_actor.NLAFSDPActor}" \
    --loss-type sft_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_actor.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$AV_SFT_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "$INSTRUCT_MODEL" \
    --save "$SAVE_DIR" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-batch-size 32 \
    --global-batch-size 32 \
    --micro-batch-size 4 \
    --lr 5e-5 --lr-decay-style constant \
    --n-samples-per-prompt 1 \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    --nla-injection-scale "$INJ_SCALE" \
    --num-epoch "${NUM_EPOCH:-1}" \
    --save-interval "${SAVE_INTERVAL:-500}" \
    "$@"
