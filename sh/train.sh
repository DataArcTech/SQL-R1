export WANDB_API_KEY=your_wandb_api_key
export VLLM_ATTENTION_BACKEND=XFORMERS

DATA_DIR_PATH=data

RUN_ID=7B
GPU_ENV=8GPU
MODEL_ENV=Qwen2.5-Coder-7B-Instruct
PROJECT_NAME=SQL-R1
        
LOG_PATH=logs/$PROJECT_NAME
MODEL_PATH=models/$MODEL_ENV
EXPERIMENT_NAME=$GPU_ENV-$MODEL_ENV-$RUN_ID

mkdir -p $LOG_PATH

set -x

nvidia-smi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR_PATH/train.parquet \
    data.val_files=$DATA_DIR_PATH/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=80 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$LOG_PATH/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=10 $@ 2>&1 | tee $LOG_PATH/$MODEL_ENV/grpo.log
