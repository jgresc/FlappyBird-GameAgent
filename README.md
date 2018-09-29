# Flappy Bird played by Reinforcement Learning Algorithms

<img src="https://raw.githubusercontent.com/jgresc/FlappyBird-GameAgent/master/assets/flappyDemo.gif" width="150">

## Overview & Disclaimer
This project was created as an assignment for a lecture at the UZH. A group of 4 students were involved. The project uses two different reinforcement learning algorithms, both of which are applied to the game Flappy Bird. The reinforcement learning algorithms were taken from other existing Repos and merged into one project and the model was re-trained.

## Algorithms
- Asynchronous Actor-Critic Agents (A3C)
- Deep Q-Network

## Installation Dependencies:
* Python 3.5
* pygame
* Keras 2.0
* scikit-image
* TensorFlow
* OpenCV-Python
* h5py

## How to Run?
![samlpe](https://github.com/jgresc/FlappyBird-GameAgent/blob/master/assets/Start.jpg?raw=true)
```
git clone https://github.com/jgresc/FlappyBird-GameAgent.git
cd RLGameAgents
python run.py
```
## Train Results
Different hyperparameters were tried out, such as various activation functions for the A3C algorithm.
![sample](https://github.com/jgresc/FlappyBird-GameAgent/blob/master/assets/trainA3C.jpg?raw=true)

The DQN algorithm was less efficient, and took more time to produce decent results.
![sample](https://raw.githubusercontent.com/jgresc/FlappyBird-GameAgent/master/assets/trainDQN.jpg)


However, it also turned out that the trained DQN can play FlappyBird better than the A3C on average.
![sample](https://github.com/jgresc/FlappyBird-GameAgent/blob/master/assets/score_A3C.jpg?raw=true | width=400)
![sample](https://github.com/jgresc/FlappyBird-GameAgent/blob/master/assets/score_DQN.jpg?raw=true | width=400)
