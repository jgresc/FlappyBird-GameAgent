import sys

import numpy as np

sys.path.append("game/")

import pygame
import wrapped_flappy_bird as game

import skimage
from skimage import transform, color, exposure

from keras.models import load_model
import keras.backend as K

BETA = 0.01
const = 1e-5


# loss function for policy output
def logloss(y_true, y_pred):  # policy loss
    return -K.sum(K.log(y_true * y_pred + (1 - y_true) * (1 - y_pred) + const), axis=-1)


# loss function for critic output
def sumofsquares(y_true, y_pred):  # critic loss
    return K.sum(K.square(y_pred - y_true), axis=-1)


def preprocess(image):
    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image, (85, 84), mode='constant')
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    return image


def playGame():
    model = load_model("a3c_saved_models/Episodes10000",
                       custom_objects={'logloss': logloss, 'sumofsquares': sumofsquares})
    game_state = game.GameState(30)

    currentScore = 0
    topScore = 0
    counter = 0
    a_t = [1, 0]
    FIRST_FRAME = True

    terminal = False
    r_t = 0
    while True:
        if FIRST_FRAME:
            x_t = game_state.getCurrentFrame()
            x_t = preprocess(x_t)
            s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=3)
            FIRST_FRAME = False
        else:
            x_t, r_t, terminal = game_state.frame_step(a_t)
            x_t = preprocess(x_t)
            s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

        y = model.predict(s_t)
        no = np.random.rand()
        a_t = [0, 1] if no < y[0] else [1, 0]  # stochastic policy
        # a_t = [0,1] if 0.5 < y[0] else [1,0]   #deterministic policy

        if r_t == 1:
            currentScore += 1
            topScore = max(topScore, currentScore)
        if terminal:
            FIRST_FRAME = True
            terminal = False

            # report performance of loaded model
            # f = open("logs/a3c_trained_scores.txt", "a")
            # f.write(str(counter) + ";" + str(currentScore) + "\n")
            # f.close()

            currentScore = 0
            counter += 1

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()


def main():
    playGame()


if __name__ == "__main__":
    main()
