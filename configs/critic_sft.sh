#!/bin/bash
# Critic SL — train the encoder to predict activations from explanation text.
# Single training group, model is NLACriticModel.
#
# Prerequisite: run nla.scripts.prepare_critic_checkpoint first to produce
# CRITIC_INIT_CKPT — a truncated-K-layer checkpoint with config.json +
# nla_meta.yaml. from_pretrained then loads K layers naturally (no arg needed).

: "${AR_SFT_PARQUET:?set AR_SFT_PARQUET to the Stage 3b parquet path}"
: "${CRITIC_INIT_CKPT:?set CRITIC_INIT_CKPT to prepare_critic_checkpoint.py output dir}"
: "${SAVE_DIR:?set SAVE_DIR for output}"

${PYTHON:-python} train.py \
    --train-backend "${TRAIN_BACKEND:-fsdp}" \
    --custom-actor-cls-path "${ACTOR_CLS:-nla.train_actor.NLAFSDPActor}" \
    --nla-model-is-critic \
    --loss-type custom_loss \
    --custom-loss-function-path nla.loss.nla_critic_loss \
    --debug-train-only \
    --disable-compute-advantages-and-returns \
    --rollout-function-path nla.rollout.sft_critic.generate_rollout \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$AR_SFT_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "$CRITIC_INIT_CKPT" \
    --save "$SAVE_DIR" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-batch-size 64 \
    --global-batch-size 64 \
    --micro-batch-size 8 \
    --lr 1e-4 --lr-decay-style constant \
    --n-samples-per-prompt 1 \
    --num-epoch "${NUM_EPOCH:-1}" \
    --save-interval "${SAVE_INTERVAL:-500}" \
    "$@"
