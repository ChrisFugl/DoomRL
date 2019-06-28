from baselines.common.tf_util import get_session
from baselines.ppo2.ppo2 import learn
import tensorflow as tf


def train(config, env):
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    session_config.gpu_options.allow_growth = True
    get_session(config=session_config)
    model = learn(
        env=env,
        total_timesteps=config.timesteps,
        network='cnn',
        seed=config.seed,
        ent_coef=config.entropy_weight,  # default = 0
        lr=config.learning_rate,
        vf_coef=config.critic_weight,  # default = 0.5
        max_grad_norm=config.max_grad_norm,
        gamma=config.discount_factor,
        nsteps=config.number_of_steps,
        eval_env=None,
        lam=0.95,
        log_interval=6,
        nminibatches=4,
        noptepochs=4,
        cliprange=0.2,
        save_interval=0,
        load_path=None,
        model_fn=None,
        update_fn=None,
        init_fn=None,
        mpi_rank_weight=1,
        comm=None
    )

    model.save(config.save_path)
