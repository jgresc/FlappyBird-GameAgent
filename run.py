import sys
sys.path.append("game/")

import pygame
import wrapped_flappy_bird as game
import deep_q_network
import a3c

pygame.init()

SCREENWIDTH = 288
SCREENHEIGHT = 405
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')
gameIcon = pygame.image.load('assets/icon.png')
pygame.display.set_icon(gameIcon)

WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 200, 0)
BRIGHT_RED = (255, 0, 0)
BRIGHT_GREEN = (0, 255, 0)
BRIGHT_BLUE = (0, 0, 255)


def text_objects(text, font):
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()


def button(msg, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(SCREEN, ac, (x, y, w, h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(SCREEN, ic, (x, y, w, h))
    smallText = pygame.font.SysFont("comicsansms", 20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    SCREEN.blit(textSurf, textRect)


def runA3CAgent():
    a3c.main()
    pygame.quit()


def runDQNAgent():
    deep_q_network.main()
    pygame.quit()


def playOnYourOwn():
    game.playGameOnYourOwn()
    pygame.quit()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()

    background = pygame.image.load('assets/sprites/background-day.png')
    SCREEN.blit(background, (0, 0))

    largeText = pygame.font.SysFont("comicsansms", 35)
    TextSurf, TextRect = text_objects("RLGameAgents", largeText)
    TextRect.center = ((SCREENWIDTH / 2), 50)
    SCREEN.blit(TextSurf, TextRect)

    smallText = pygame.font.SysFont("comicsansms", 10)
    TextSurf, TextRect = text_objects("HS17 Reinforcement Learning for Economists and Managers", smallText)
    TextRect.center = ((SCREENWIDTH / 2), 100)
    SCREEN.blit(TextSurf, TextRect)

    mediumText = pygame.font.SysFont("comicsansms", 20)
    TextSurf, TextRect = text_objects("Choose Game Mode:", mediumText)
    TextRect.center = ((SCREENWIDTH / 2), 160)
    SCREEN.blit(TextSurf, TextRect)

    button("A3C", (SCREENWIDTH / 2) - 80, 180, 160, 50, GREEN, BRIGHT_GREEN, runA3CAgent)
    button("Deep QLearning", (SCREENWIDTH / 2) - 80, 250, 160, 50, RED, BRIGHT_RED, runDQNAgent)
    button("Play on your own", (SCREENWIDTH / 2) - 80, 320, 160, 50, BLUE, BRIGHT_BLUE, playOnYourOwn)

    pygame.display.update()
