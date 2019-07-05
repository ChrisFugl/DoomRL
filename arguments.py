import configargparse
import os


def parse_test_config():
    parser = configargparse.get_arg_parser()
    parser.add(
        '-a', '--algorithm',
        choices=['baseline_a2c', 'baseline_ppo', 'a2c', 'ppo'],
        required=True,
        help='Algorithm to use. One of: baseline_a2c, a2c, baseline_ppo2.'
    )
    parser.add('-e', '--env', required=True, help='Name of Vizdoom environment. See README for a list of environment names.')
    parser.add('-n', '--name', required=True, help='Name of the model (as set during training).')
    parser.add('-o', '--out', required=True, help='Save path of experiments.')
    parser.add('-sv', '--save_video', action='store_true', help='Save videos of rollouts (default false).')
    parser.add('-s', '--seed', required=False, type=int, default=None, help='Random seed.')
    parser.add('-r', '--rollouts', required=False, type=int, default=100, help='Rollouts per experiment (default 100).')
    parser.add('-dr', '--downscale_ratio', required=False, type=int, default=1.0, help='Down scale ratio (default 1 - no downscaling.)')
    parser.add('-g', '--grayscale', action='store_true', help='Use grayscale (default false).')
    parser.add('-fs', '--framestacking', required=False, type=int, default=0, help='Number of stacked frames (default 0 - do not stack).')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    file_path = os.path.dirname(os.path.realpath(__file__))
    load_path = os.path.join(file_path, 'out')
    args.load_path = os.path.join(load_path, args.name, 'model')
    args.video_path = os.path.join(args.out, 'videos')

    # set dummy training variables that are needed to initialize models but will not actually be used
    args.timesteps = 0
    args.learning_rate = 0
    args.discount_factor = 0
    args.number_of_steps = 1
    args.number_of_environments = 1
    args.batch_size = 1
    args.mini_batch_size = 1
    args.epochs = 1
    args.gae_lambda = 0
    args.clip_epsilon = 0
    args.momentum = 0
    args.rmsp_decay = 0
    args.rmsp_epsilon = 0
    args.entropy_weight = 0
    args.critic_weight = 0
    args.max_grad_norm = 0
    args.sampling_method = 'max'
    args.epsilon = 0
    args.reward_scale = 1

    return args


def parse_train_config():
    parser = configargparse.get_arg_parser()
    parser.add(
        '-a', '--algorithm',
        choices=['baseline_a2c', 'baseline_ppo', 'a2c', 'ppo'],
        required=True,
        help='Algorithm to use. One of: baseline_a2c, a2c, baseline_ppo2.'
    )
    parser.add('-e', '--env', required=True, help='Name of Vizdoom environment. See README for a list of environment names.')
    parser.add('-n', '--name', required=True, help='Name of experiment - used to generate log and output files.')
    parser.add('-t', '--timesteps', required=False, type=int, default=1000000, help='Number of timesteps (default 1 million)')
    parser.add('-lr', '--learning_rate', required=False, type=float, default=1e-4, help='Learning rate (default 0.0001).')
    parser.add('-s', '--seed', required=False, type=int, default=None, help='Random seed.')
    parser.add('-d', '--discount_factor', required=False, type=float, default=0.99, help='Discount factor (default 0.99).')
    parser.add('-ns', '--number_of_steps', required=False, type=int, default=5, help='Number of steps (default 5).')
    parser.add('-ne', '--number_of_environments', required=False, type=int, default=8, help='Number of environments in A2C (default 8).')
    parser.add('-mbs', '--mini_batch_size', required=False, type=int, default=128, help='Mini batch size in PPO (default 128).')
    parser.add('-epochs', '--epochs', required=False, default=4, help='Epochs of training in PPO (default 4).')
    parser.add('-m', '--momentum', required=False, type=float, default=0.0, help='Optimizer momentum (default 0).')
    parser.add('-rmspd', '--rmsp_decay', required=False, type=float, default=0.99, help='RMSprop decay (default 0.99).')
    parser.add('-rmspe', '--rmsp_epsilon', required=False, type=float, default=1e-10, help='RMSprop epsilon (default 1e-10).')
    parser.add('-sc', '--skipcount', required=False, type=int, default=1, help='Number of frames to skip (default 1).')
    parser.add('-dr', '--downscale_ratio', required=False, type=int, default=1.0, help='Down scale ratio (default 1 - no downscaling.)')
    parser.add('-g', '--grayscale', action='store_true', help='Use grayscale (default false).')
    parser.add('-fs', '--framestacking', required=False, type=int, default=0, help='Number of stacked frames (default 0 - do not stack).')
    parser.add('-sv', '--save_video', action='store_true', help='Save videos while training (default false).')
    parser.add('-ew', '--entropy_weight', required=False, type=float, default=0.01, help='Weight of entropy (default 0.01).')
    parser.add('-cw', '--critic_weight', required=False, type=float, default=0.5, help='Wegith of critic loss (default 0.5).')
    parser.add(
        '-mgn',
        '--max_grad_norm',
        required=False,
        type=float,
        default=None,
        help='Clips gradients at this norm or higher (default no clipping).'
    )
    parser.add(
        '-sm',
        '--sampling_method',
        choices=['noise', 'categorical', 'epsilon', 'max'],
        required=False,
        default='noise',
        help='Method used for sampling (default noise).'
    )
    parser.add('-ep', '--epsilon', required=False, type=float, default=0.1, help='Epsilon in greedy epsilon sampling (default 0.1).')
    parser.add('-le', '--log_every', required=False, type=int, default=100, help='Log every nth update (default 100).')
    parser.add('-ce', '--clip_epsilon', required=False, type=float, default=0.2, help='Epsilon in clip in PPO (default 0.2).')
    parser.add(
        '-gael',
        '--gae_lambda',
        required=False,
        type=float,
        default=0.95,
        help='"Generalized Advantage Estimation" lambda (default 0.95).'
    )
    parser.add('-rs', '--reward_scale', required=False, type=float, default=1, help='Downscale reward by a constant (default 1).')
    args = parser.parse_args()
    args.batch_size = args.number_of_steps * args.number_of_environments
    file_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(file_path, 'out')
    log_path = os.path.join(file_path, 'logs')
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    args.save_path = os.path.join(out_path, args.name, 'model')
    args.video_path = os.path.join(out_path, args.name)
    args.log_path = os.path.join(log_path, args.name)
    args.stopping_criterion = get_stopping_criterion(args.timesteps)
    return args


def get_stopping_criterion(timesteps):

    def stop_after_timestep(timestep):
        return timesteps < timestep

    return stop_after_timestep
