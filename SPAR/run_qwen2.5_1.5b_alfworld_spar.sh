#!/bin/bash
set -euo pipefail
set -x

ENGINE=${1:-vllm}
if [[ $# -gt 0 ]]; then shift; fi

export PYTHONPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH:-}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/workspace/verl-agent/alfworld_data}
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
DATA_DIR=${DATA_DIR:-/workspace/verl-agent}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-./checkpoints}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-spar_qwen2.5_1.5b_alfworld_seed0}
MAX_STEPS=${MAX_STEPS:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "$DATA_DIR" \
    --train_data_size 16 \
    --val_data_size 128

python3 -m SPAR.main_spar \
    algorithm.adv_estimator=spar \
    algorithm.gamma=0.10 \
    algorithm.spar.step_advantage_w=1.0 \
    algorithm.spar.episode_advantage_w=1.0 \
    algorithm.spar.mode=mean_std_norm \
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
    actor_rollout_ref.model.use_remove_padding=True \
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
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
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
    trainer.logger=['console'] \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$CHECKPOINTS_DIR/$EXPERIMENT_NAME \
    "$@"
