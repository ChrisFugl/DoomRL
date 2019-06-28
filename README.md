# DoomRL
Deep reinforcement learning algorithms in a Doom environment based on [OpenAI baselines](https://github.com/openai/baselines).

## Usage
Use train.py to train a model.

```
python3 train.py -a a2c -e VizdoomBasic-v0 -n my_model
```

Enter `python3 train.py --help` to get a list of available parameters for training. The environment requires the presence of a screen. This may be an issue if you are running this on a server. In this case you should train using `sh train_with_xfvb.sh` instead of `python3 train.py`.

Logs are automatically written to the `logs` directory and can be visualized using tensorboard.
```
tensorboard --logdir logs
```

Once a model has finished training, you can test its performance using `test.py` which generates a number of rollouts and can be configured to save a video of each rollout.

```
python3 test.py -a a2c -e VizdoomBasic-v0 -n my_model -o path/to/performance/results
```

Enter `python3 test.py --help` to get a list of available parameters for testing. Similar to training, use `sh test_with_xvfb.sh` if you do not have access to a physical screen.

### Scenarios
The following doom scenarios are available. A description of each scenario can be seen on [this page](https://github.com/shakenes/vizdoomgym/blob/master/vizdoomgym/envs/scenarios/README.md).

```
VizdoomBasic-v0
VizdoomCorridor-v0
VizdoomDefendCenter-v0
VizdoomDefendLine-v0
VizdoomHealthGathering-v0
VizdoomMyWayHome-v0
VizdoomPredictPosition-v0
VizdoomTakeCover-v0
VizdoomDeathmatch-v0
VizdoomHealthGatheringSupreme-v0
```

## Requirements
Python 3.6+.

```
pip3 install --user -r requirements.txt
```

Install [baselines](https://github.com/openai/baselines).

Install Vizdoom. See [this page](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#pypi) for installation instructions.

## Acknowledgements
Models are based on implementations by [OpenAI baselines](https://github.com/openai/baselines). Code in vizdoomgym directory is based of the work in https://github.com/shakenes/vizdoomgym by Simon Hakenes.
