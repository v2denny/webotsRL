import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
from controller import Robot, TouchSensor, DistanceSensor, Supervisor, GPS, Compass
from scripts.utils import cmd_vel
import math
from typing import Union, Dict, List, Tuple
import numpy as np
from scripts.positions import get_positions
from scripts.utils import warp_robot
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
DIST_THRESHOLD = 0.1


class WebotsEnv(gym.Env):


    def __init__(self):
        super(WebotsEnv, self).__init__()
        self.supervisor: Supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())
        self.lidar = self.supervisor.getDevice('lidar')
        self.lidar.enablePointCloud()
        self.lidar.enable(timestep)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0] * 9 + [0, -np.pi]),
            high=np.array([1] * 9 + [1, np.pi]),
            dtype=np.float32)
        self.touch_sensor: TouchSensor = self.supervisor.getDevice('touch sensor')
        self.touch_sensor.enable(timestep)
        self.gps: GPS = self.supervisor.getDevice('gps')
        self.gps.enable(timestep)
        self.compass: Compass = self.supervisor.getDevice('compass')
        self.compass.enable(timestep)
        
        self.trainmode = "all"
        # start pos = only one position + towards target
        # easy pos = easypos + towards target
        # medium pos = easypos + rnd direction target
        # hard pos = hardpos + towards target
        # all pos = hardpos + rnd direction target (all maps)
        
        self.robotpos, self.targetpos = get_positions(str(self.trainmode))
        print("robotpos:",self.robotpos)
        print("targetpos:",self.targetpos)
        if (len(self.targetpos) > 1): self.targs = 2
        else: self.targs = 1
        self.steps = 0  # To keep track of steps per episode
        self.max_steps_per_episode = 800  # Limit to prevent infinite loops

        # Track visited locations
        self.visited_locations = set()
        self.visit_threshold = 0.1  # Minimum distance to consider as a new location

        # Incentivate movement
        self.supervisor.step()
        self.previous_position = self.gps.getValues()[:2]


    def reset(self, seed=None, options=None):
        # Reset counters
        self.visited_locations = set()
        self.steps = 0
        self.prev_distance_to_target = self.calculate_distance_to_target()[0]
        self.same_spot_steps = 0

        # Start on a random position
        if seed is not None:
            np.random.seed(seed)
        self.robotpos, self.targetpos = get_positions(str(self.trainmode))

        if (len(self.targetpos) > 1): self.targs = 2
        else: self.targs = 1
        xpos, ypos = self.robotpos
        warp_robot(self.supervisor, "EPUCK", (xpos, ypos))

        # Rotate robot to target
        if str(self.trainmode) in ["start", "easy", "hard"]:
            self.rotate_to_target()

        return self.get_observation(), {}


    def step(self, action):
        self.steps += 1

        # If colision detected return negative reward
        done = self.colision()
        if done == True :
            return self.get_observation(), -100, True, False, {}

        # Choose the action
        if action == 0:
            cmd_vel(self.supervisor, 0.12, 0)
        elif action == 1:
            cmd_vel(self.supervisor, 0, 1.3)
        elif action == 2:
            cmd_vel(self.supervisor, 0, -1.3)
        elif action == 3:
            cmd_vel(self.supervisor, -0.06, 0)
        self.supervisor.step(200)

        done = self.colision()
        if done == True :
            return self.get_observation(), -100, True, False, {}

        # Truncate episode
        truncated = self.steps >= self.max_steps_per_episode

        reward, done = self.calculate_reward_done_v6()
        return self.get_observation(), reward, done, truncated, {}


    def colision(self):
        self.supervisor.step()
        if self.touch_sensor.getValue() == 1.0:
            return True
        else: return False


    def rotate_to_target(self):
        obs = self.get_observation()
        ang = float(obs[-1])
        while abs(ang) >= 0.1:
            cmd_vel(self.supervisor, 0, 2)
            self.supervisor.step()
            obs = self.get_observation()
            ang = float(obs[-1])
        cmd_vel(self.supervisor, 0, 0)
        self.supervisor.step(1)


    def get_observation(self):
        self.supervisor.step()
        point_cloud = np.array(self.lidar.getPointCloud())
        lidar_features = self.process_lidar_v2(point_cloud)
        distance, nearest_target = self.calculate_distance_to_target()
        angle = self.calculate_angle_to_target(nearest_target)
        observations = np.concatenate([lidar_features, [distance, angle]])
        return observations
    

    def calculate_reward_done_v6(self):
        self.supervisor.step()
        distance, closest_target = self.calculate_distance_to_target()

        # Collision penalty
        if self.touch_sensor.getValue() == 1.0:
            done = True
            reward = -100
            return reward, done

        # Reached the target
        if distance < DIST_THRESHOLD and self.targs == 1:
            done = True
            reward = 100
        elif distance < DIST_THRESHOLD and self.targs == 2:
            self.targs = 1
            done = False
            reward = 80
            self.targetpos.remove(closest_target)
            print("removed closest target: self.targetpos len =", len(self.targetpos))
        else:
            done = False
            reward = -distance*0.5  # Negative reward based on distance to the target

        # Additional reward shaping
        reward += 5 * (self.prev_distance_to_target - distance)  # Reward for getting closer
        reward -= 0.8  # Small negative reward for each step taken (time penalty)

        # Check if the current position is a new location
        gps_readings: List[float] = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        current_position = (round(robot_position[0], 1), round(robot_position[1], 1))
        if current_position not in self.visited_locations:
            self.visited_locations.add(current_position)
            reward += 2  # Reward for visiting a new location

        distance_moved = np.linalg.norm(np.array(robot_position) - np.array(self.previous_position))
        reward += distance_moved * 2  # Reward for movement
        self.previous_position = robot_position  # Update the previous position

        # Proximity to wall penalty
        point_cloud = np.array(self.lidar.getPointCloud())
        lidar_features = self.process_lidar_v2(point_cloud)
        for feature in lidar_features:
            if feature <= 0.02: reward -= 0.75

        # Penalty for staying in the same spot (spinning)
        if distance >= self.prev_distance_to_target:
            self.same_spot_steps += 1
        else:
            self.same_spot_steps = 0

        if self.same_spot_steps > 10:  # Threshold for steps in the same spot
            reward -= 40  # Penalty for staying in the same spot
            self.same_spot_steps = 0

        # Update previous distance
        self.prev_distance_to_target = distance

        return reward, done


    def calculate_distance_to_target(self):
        dismin = 100000
        mintargetpos = None
        for targetpos in self.targetpos:
            (xtarget, ytarget) = targetpos
            self.supervisor.step()
            gps_readings: List[float] = self.gps.getValues()
            robot_position = (gps_readings[0], gps_readings[1])
            distance = math.sqrt((robot_position[0] - xtarget) ** 2 + (robot_position[1] - ytarget) ** 2)
            if distance < dismin:
                dismin = distance
                mintargetpos = targetpos
        return dismin, mintargetpos
    

    # Calculate the orientation angle of the robot
    def get_robot_orientation(self) -> float:
        self.supervisor.step()
        compass_values = self.compass.getValues()
        orientation = math.atan2(compass_values[0], compass_values[1])
        return orientation

    # Get current GPS readings and robot orientation
    def calculate_angle_to_target(self, target_pos) -> float:
        self.supervisor.step()
        gps_readings: List[float] = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        robot_orientation = self.get_robot_orientation()

        target_vector = (target_pos[0] - robot_position[0], target_pos[1] - robot_position[1])
        angle_to_target = math.atan2(target_vector[1], target_vector[0])
        relative_angle = angle_to_target - robot_orientation
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

        return relative_angle


    def process_lidar_v2(self, point_cloud):
        distances = []

        # Select 9 points from the lidar readings (100 total)
        ids = [0, 11, 24, 36, 49, 61, 74, 86, 99]
        pc9 = [point_cloud[i] for i in ids]

        for point in pc9:
            x = point.x
            y = point.y
            distance = np.sqrt(x ** 2 + y ** 2)
            distances.append(min(distance, 2))

        # Normalize to range [0, 1]
        distances = np.array(distances)
        distances = np.clip(distances / 2, 0, 1)
        return distances


    def render(self, mode='human'):
        pass


env = WebotsEnv()
model_path = 'none'
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./', name_prefix='ppov6')
if os.path.exists(model_path):
    model = PPO.load(model_path, env=env)
    print("Model loaded from", model_path)
else:
    model = PPO('MlpPolicy', env, verbose=1)
    print("Training a new model")
model.learn(total_timesteps=1000000, callback=checkpoint_callback)
model.save("FINAL")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, truncated, _ = env.step(action)
    if dones or truncated:
        obs, _ = env.reset()

