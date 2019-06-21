# DoomRL
Hi.

## Usage
Use train.py to train stuff.

Important to prefix with `xvfb-run -s "-screen 0 320x240x24"` if running from a server. This can also be done by running `sh train_with_xfvb`.

Use tensorboard to visualize training:

```
tensorboard --logdir logs
```

### Environments
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
Python 3.6 and see requirements.txt.

```
pip3 install --user -r requirements.txt
```

Install baselines.
https://github.com/openai/baselines

Vizdoom.
See this page for installation instructions: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#pypi

## Acknowledgements
Code in vizdoomgym directory is a slight modification of the work in https://github.com/shakenes/vizdoomgym by Simon Hakenes.
