import pygame
import numpy as np
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import os
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from ai import FirmEnvironment, get_action

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = (400, 400)
NUM_FIRMS = 10
SQUARE_SIZE = SCREEN_SIZE[0] // NUM_FIRMS

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Create the screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("AI Firm Actions Visualization")


def draw_firm_actions(actions, productivity):
    screen.fill(WHITE)

    for i in range(NUM_FIRMS):
        color = (
            RED
            if int(actions[i]) == 1 and productivity[i] <= env.avg_productivity
            else GREEN
        )
        x = i * SQUARE_SIZE
        pygame.draw.rect(screen, color, (x, 0, SQUARE_SIZE, SQUARE_SIZE))

    pygame.display.flip()
    time.sleep(2)


# Create an instance of the FirmEnvironment class
env = FirmEnvironment(num_firms=10, num_periods=10)

# Loading the model
# Load the trained model using SavedModel format
loaded_model = tf.keras.models.load_model("trained_model")

# Compile the loaded model with the same configuration as during training
optimizer = Adam(
    learning_rate=0.01
)  # Use the same optimizer configuration as during training
loaded_model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

loaded_model.summary()
# Load individual firm models using SavedModel format
loaded_firm_models = []
for firm_index in range(env.num_firms):
    firm_model = tf.keras.models.load_model(f"firm_model_{firm_index}")
    firm_model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    loaded_firm_models.append(firm_model)


# Main loop
running = True
period = 0
episode_results = []
while running:
    if period != env.num_periods:
        out_of_market_counts = 0
        action_list = []
        state = env.endowments.copy()

        for firm_index in range(env.num_firms):
            action = get_action(state[firm_index], env.avg_productivity)
            action_list.append(action)

        rewards, next_state = env.step(action_list)
        episode_results.append((rewards, env.avg_productivity))

        for firm_index in range(env.num_firms):
            if action_list[firm_index] == 0:
                out_of_market_counts += 1

        # Calculate share of firms staying out of the market
        share_out_of_market = out_of_market_counts / env.num_firms
        print(
            f"Period: {period}, Share of firms out of market: {share_out_of_market:.2f}"
        )
    else:
        running = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_firm_actions(action_list, env.productivity)

    period += 1

# Clean up
pygame.quit()
