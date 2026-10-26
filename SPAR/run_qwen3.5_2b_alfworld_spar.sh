#!/bin/bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then shift; fi

export PYTHONPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}
PYTHON=${PYTHON:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv-qwen35/bin/python}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-2B}
DATA_DIR=${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-./checkpoints}
SUBGOAL_PROMPT=${SUBGOAL_PROMPT:-default}
GRAPH_BACKEND=${GRAPH_BACKEND:-graphgpo}  # graphgpo | rewardflow
EXPERIMENT_NAME=${EXPERIMENT_NAME:-spar_${GRAPH_BACKEND}_qwen3.5_2b_alfworld_${SUBGOAL_PROMPT}_seed0}
MAX_STEPS=${MAX_STEPS:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

"$PYTHON" -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$DATA_DIR" \
    --train_data_size 16 \
    --val_data_size 128

"$PYTHON" -m SPAR.main_spar \
    algorithm.adv_estimator=spar \
    algorithm.gamma=0.10 \
    algorithm.spar.step_advantage_w=1.0 \
    algorithm.spar.episode_advantage_w=1.0 \
    algorithm.spar.mode=mean_std_norm \
    algorithm.spar.subgoal_prompt=$SUBGOAL_PROMPT \
    algorithm.spar.graph_backend=$GRAPH_BACKEND \
    algorithm.use_kl_in_reward=False \
    data.train_files=$DATA_DIR/text/train.parquet \
    data.val_files=$DATA_DIR/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.text_only=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT:-aaai_rebuttal} \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-10} \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$CHECKPOINTS_DIR/$EXPERIMENT_NAME \
    "$@"
