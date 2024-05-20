import gymnasium as gym
from gymnasium import spaces
from controller import Robot, TouchSensor, DistanceSensor
from controllers.utils import cmd_vel
from webotsRL.scripts.positions import next_pos, print_pos
import math
from typing import Union, Dict,List,Tuple
import numpy as np
from positions import get_positions


DIST_THRESHOLD = 0.1

class WebotsEnv(gym.Env):

    def __init__(self):
        super(WebotsEnv, self).__init__()
        self.supervisor: Supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())
        self.lidar = self.supervisor.getDevice('lidar')
        self.lidar.enable(timestep)
        self.action_space = spaces.Discrete(3)
        self.touch_sensor: TouchSensor = supervisor.getDevice('touch sensor')
        self.touch_sensor.enable(timestep)   
        self.gps: GPS = supervisor.getDevice('gps')
        self.gps.enable(timestep)
        self.compass: Compass = supervisor.getDevice('compass')
        self.compass.enable(timestep)
        self.robotpos = get_positions("easy")[0]
        self.targetpos = get_positions("easy")[1]

    def step(self, action):
                # Perform action
        if action == 0:
            cmd_vel(self.supervisor,1,0)
        elif action == 1:
            cmd_vel(self.supervisor,0,1)
        elif action == 2:
            cmd_vel(self.supervisor,0,-1)


        observation = self.get_observation()
        reward, done = self.calculate_reward_done()
        


        return observation, reward, done

    def reset(self):
        self.robotpos = get_positions("easy")[0]
        self.targetpos = get_positions("easy")[1]


    def get_observation(self):
        point_cloud =  np.array(self.lidar.getPointCloud())
        print(point_cloud)
        lidar_features = self.process_lidar(point_cloud)

        distance,nearest_target = self.calculate_distance_to_target()
        angle = self.calculate_angle_to_target(nearest_target)
        observations = np.concatenate([lidar_features, [distance, angle]])
        return observations


    def calculate_reward_done(self):
        if self.touch_sensor.getValue() == 1.0:
            done = True
            reward = -1
        if self.calculate_distance_to_target() < DIST_THRESHOLD:
            done = True
            reward = 1
        else:
            done = False
            reward = 0
        return reward, done

    def calculate_distance_to_target(self): #função retorna distância do robo ao alvo mais próximo
        dismin = math.inf
        for targetpos in self.targetpos:
            (xtarget,ytarget) = targetpos
            gps_readings: List[float] = self.gps.getValues()
            robot_position = (gps_readings[0], gps_readings[1])
            distance = math.sqrt((robot_position[0] - xtarget)**2 + (robot_position[1] - ytarget)**2)
            if distance < dismin:
                dismin = distance
                mintargetpos = targetpos
                print("dismin",dismin)
                print("mintargetpos",mintargetpos)
        return dismin, mintargetpos
    

    def get_robot_orientation(self) -> float:
        # Assuming compass.getValues() returns values such as [x, y, z]
        compass_values = self.compass.getValues()
        # Calculate the orientation angle of the robot
        orientation = math.atan2(compass_values[0], compass_values[1])
        return orientation

    def calculate_angle_to_target(self, target_pos) -> float:
        # Get current GPS readings and robot orientation
        gps_readings: List[float] = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        robot_orientation = self.get_robot_orientation()

        # Calculate the vector to the target from the robot
        target_vector = (target_pos[0] - robot_position[0], target_pos[1] - robot_position[1])
        
        # Calculate the angle from the robot to the target
        angle_to_target = math.atan2(target_vector[1], target_vector[0])

        # Calculate the relative angle from the robot's forward facing direction to the target
        relative_angle = angle_to_target - robot_orientation

        #ver qual o melhor angulo
        #relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi #Normalize between -pi and pi
        relative_angle = (relative_angle + np.pi) / (2 * np.pi)  # Normalize between 0 and 1

        return relative_angle


    def process_lidar(self, point_cloud):
        # Example: Convert point cloud to polar coordinates and take min distance per sector
        num_sectors = 8
        sector_width = 360 / num_sectors
        min_distances = np.full(num_sectors, np.inf)
        
        for point in point_cloud:
            angle = np.degrees(np.arctan2(point[1], point[0]))
            sector = int(angle // sector_width)
            distance = np.sqrt(point[0]**2 + point[1]**2)
            min_distances[sector] = min(min_distances[sector], distance)

        return min_distances