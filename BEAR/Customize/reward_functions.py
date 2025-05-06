import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from numpy import linalg as LA
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import inv

def my_custom_reward_function3(self, state, action, error, state_new):
    # This is your default reward function
    # Initialize the reward
    reward = 0
    self.beta=0.8



    # Calculate the contribution of action to the reward
    action_contribution = LA.norm(action, 2) * (1-self.beta)
    reward -= action_contribution

    # Calculate the contribution of error to the reward
    error_contribution = LA.norm(error, 2) * self.beta
    reward -= error_contribution

    return reward

def my_custom_reward_function2(self, state, action, error, state_new):
    # Initialize the reward
    reward = 0
    self.co2_rate = 0.01
    self.temp_rate = 0.5

    # Desired temperature range
    standard_lower_temp = self.target - 1
    standard_upper_temp = self.target + 1

    # Increased deadband for unoccupied zones
    increased_deadband = 2
    lower_temp_unoccupied = self.target - increased_deadband
    upper_temp_unoccupied = self.target + increased_deadband

    # Occupancy information
    occ_info = state[-self.roomnum:]  # Assumes the last 'roomnum' elements are occupancy info

    # Adjust temperature bounds based on occupancy
    lower_temp = np.where(occ_info > 0, standard_lower_temp, lower_temp_unoccupied)
    upper_temp = np.where(occ_info > 0, standard_upper_temp, upper_temp_unoccupied)

    # Calculate temperature deviations
    temp_dev_array = np.maximum(0, state_new[:self.roomnum] - upper_temp) + np.maximum(0, lower_temp - state_new[:self.roomnum])
    temp_deviation = np.sum(temp_dev_array) * self.temp_rate

    # Subtract temperature deviation from the reward
    reward -= temp_deviation

    # Calculate the contribution of CO2 emissions to the reward
    co2_emission = LA.norm(action, 2) * self.co2_rate
    reward -= co2_emission

    # Print detailed debug information if necessary
    #print('Temperature deviation:', temp_deviation)
    #print('CO2 emission:', co2_emission)
    #print('Reward:', reward)
    #print("reward function 2")
    return reward

def my_custom_reward_function1(self, state, action, error, state_new):
    # This is your default reward function
    # Initialize the reward
    reward = 0
    self.co2_rate=0.01
    self.temp_rate=0.01

    # Desired temperature range
    lower_temp = 18
    upper_temp = 22

    # Calculate the contribution of action to the reward
    action_contribution = LA.norm(action, 2) * self.q_rate
    reward -= action_contribution

    # Calculate the contribution of error to the reward
    error_contribution = LA.norm(error, 2) * self.error_rate
    reward -= error_contribution

    # Calculate the contribution of temperature deviation to the reward
    temp_deviation = np.sum(np.maximum(0, state_new - upper_temp) + np.maximum(0, lower_temp - state_new)) * self.temp_rate
    reward -= temp_deviation

    # Calculate the contribution of CO2 emissions to the reward
    co2_emission = LA.norm(action, 2) * self.co2_rate
    reward -= co2_emission

    return reward