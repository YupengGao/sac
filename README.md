## SAC
Soft Actor-Critic implementation with Tensorflow.

https://arxiv.org/pdf/1801.01290.pdf

## requirements
- Python3

## dependencies
- tensorflow
- gym[atari]
- opencv-python
- git+https://github.com/imai-laboratory/rlsaber

## usage
### training
```
$ python train.py [--render]
```

### playing
```
$ python train.py [--render] [--load {path of models}] --demo
```

### implementation
This is inspired by following projects.

- [DQN](https://github.com/imai-laboratory/dqn)
