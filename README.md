# DoomRL
Hi.

## Usage
Use train.py to train stuff. Important to prefix with `xvfb-run -s "-screen 0 640x480x24"` if running from a server.

Use tensorboard to visualize training:

```
tensorboard --logdir logs
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
