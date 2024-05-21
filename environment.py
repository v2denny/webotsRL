import gymnasium as gym
from gymnasium import spaces
from controller import Robot, TouchSensor, DistanceSensor, Supervisor, GPS, Compass
from webotsRL.scripts.utils import cmd_vel
import math
from typing import Union, Dict, List, Tuple
import numpy as np
from webotsRL.scripts.positions import get_positions
from webotsRL.scripts.utils import warp_robot
import random
from stable_baselines3 import PPO

# easy pos = easypos+towards target
# medium pos = easypos+rnd direction target
# hard pos = hardpos+rnd direction target

DIST_THRESHOLD = 0.1



class WebotsEnv(gym.Env):

    def __init__(self):
        super(WebotsEnv, self).__init__()
        self.supervisor: Supervisor = Supervisor()
        timestep = int(self.supervisor.getBasicTimeStep())
        self.lidar = self.supervisor.getDevice('lidar')
        self.lidar.enablePointCloud()  # Enable point cloud
        self.lidar.enable(timestep)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0] * 8 + [0, 0]),  # 8 lidar sectors, distance, angle
            high=np.array([1] * 8 + [1, 1]),  # Adjust high values as needed
            dtype=np.float32
        )
        self.touch_sensor: TouchSensor = self.supervisor.getDevice('touch sensor')
        self.touch_sensor.enable(timestep)
        self.gps: GPS = self.supervisor.getDevice('gps')
        self.gps.enable(timestep)
        self.compass: Compass = self.supervisor.getDevice('compass')
        self.compass.enable(timestep)
        self.robotpos, self.targetpos = get_positions("easy")
        print("robotpos:",self.robotpos)
        print("targetpos:",self.targetpos)


        if (len(self.targetpos) > 1):
            self.targs = 2
        else:
            self.targs = 1

    def reset(self, seed=None, options=None):
        print("RESET HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        # Optionally handle the seed here if needed
        if seed is not None:
            np.random.seed(seed)
        self.robotpos, self.targetpos = get_positions("easy")
        xpos, ypos = self.robotpos
        warp_robot(self.supervisor, "EPUCK", (xpos, ypos))
        return self.get_observation(), {}


    def step(self, action):
        '''
        # If colision detected return negative reward
        done = self.colision()
        if done == True :
            return self.get_observation(), -1.0, True
        '''
        # Choose the action
        if action == 0:
            cmd_vel(self.supervisor, 0.1, 0)
        elif action == 1:
            cmd_vel(self.supervisor, 0, 0.1)
        elif action == 2:
            cmd_vel(self.supervisor, 0, -0.1)
        ''' 
        done = self.colision()
        reward = 1 if self.is_on_target() else 0
        '''
        reward, done = self.calculate_reward_done()
        return self.get_observation(), reward, done,{},{}

    '''
    def colision(self):
        if self.touch_sensor.getValue() == 1.0:
            return True
        else: return False
    '''

    def is_on_target(self):
        self.supervisor.step()
        gps_readings = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        if robot_position in self.targetpos:
            return True
        else:
            return False

    def get_observation(self):
        self.supervisor.step()
        point_cloud = np.array(self.lidar.getPointCloud())
        print(point_cloud)
        lidar_features = self.process_lidar(point_cloud)

        distance, nearest_target = self.calculate_distance_to_target()
        print("nearest_target=",nearest_target)
        angle = self.calculate_angle_to_target(nearest_target)
        observations = np.concatenate([lidar_features, [distance, angle]])
        print("observations:", observations)  # Debug print
        #
        #TO DO : TRUNCATED, QUANDO BATE, NORMALIZAR ( LIDAR - 0 A 2, DISTANCEMAX = 0 A 2.2, ANGULO DE 0 A 1 OUTRA VEZ), alterar basic time step, meter steps andtes do touch sensor, etc
        return observations

    def calculate_reward_done(self):
        self.supervisor.step()
        if self.touch_sensor.getValue() == 1.0:
            done = True
            reward = -1
            return reward,done

        distance, closest_target = self.calculate_distance_to_target()

        if distance < DIST_THRESHOLD and self.targs == 1:
            done = True
            reward = 1

        elif distance < DIST_THRESHOLD and self.targs == 2:
            self.targs = 1
            done = False
            reward = 1
            self.targetpos.remove(closest_target)
            print("removeu closest target: self.tarfetposlen =", len(self.targetpos))
        else:
            done = False
            reward = 0

        return reward, done

    def calculate_reward_done2(self):
        if self.touch_sensor.getValue() == 1.0:
            done = True
            reward = -1
        distance, closest_target = self.calculate_distance_to_target()

        if distance < DIST_THRESHOLD and self.targs == 1:
            done = True
            reward += 1
        elif distance < DIST_THRESHOLD and self.targs == 2:
            self.targs = 1
            done = False
            reward += 1
            self.targetpos.remove(closest_target)
        else:
            done = False
            reward = reward - 0.0001
        return reward, done

    def calculate_distance_to_target(self):  # função retorna distância do robo ao alvo mais próximo
        dismin = 100000
        mintargetpos = None
        for targetpos in self.targetpos:
            (xtarget, ytarget) = targetpos
            print("targetpos",targetpos)
            print("xtarget",xtarget)
            self.supervisor.step()
            gps_readings: List[float] = self.gps.getValues()
            print("gpsreadings",gps_readings)
            robot_position = (gps_readings[0], gps_readings[1])
            distance = math.sqrt((robot_position[0] - xtarget) ** 2 + (robot_position[1] - ytarget) ** 2)
            print("distancengga",distance)
            if distance < dismin:
                dismin = distance
                mintargetpos = targetpos
                print("dismin", dismin)
                print("mintargetpos", mintargetpos)
        return dismin, mintargetpos

    def get_robot_orientation(self) -> float:
        # Assuming compass.getValues() returns values such as [x, y, z]
        self.supervisor.step()
        compass_values = self.compass.getValues()
        # Calculate the orientation angle of the robot
        orientation = math.atan2(compass_values[0], compass_values[1])
        return orientation

    def calculate_angle_to_target(self, target_pos) -> float:
        # Get current GPS readings and robot orientation
        self.supervisor.step()
        gps_readings: List[float] = self.gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        robot_orientation = self.get_robot_orientation()

        # Calculate the vector to the target from the robot
        target_vector = (target_pos[0] - robot_position[0], target_pos[1] - robot_position[1])

        # Calculate the angle from the robot to the target
        angle_to_target = math.atan2(target_vector[1], target_vector[0])

        # Calculate the relative angle from the robot's forward facing direction to the target
        relative_angle = angle_to_target - robot_orientation

        # ver qual o melhor angulo
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi #Normalize between -pi and pi
        #relative_angle = (relative_angle + np.pi) / (2 * np.pi)  # Normalize between 0 and 1

        return relative_angle

    '''def process_lidar(self, point_cloud):
        # Example: Convert point cloud to polar coordinates and take min distance per sector
        num_sectors = 8
        sector_width = 360 / num_sectors
        min_distances = np.full(num_sectors, np.inf)

        for point in point_cloud:
            angle = np.degrees(np.arctan2(point[1], point[0]))
            sector = int(angle // sector_width)
            distance = np.sqrt(point[0] ** 2 + point[1] ** 2)
            min_distances[sector] = min(min_distances[sector], distance)

        return min_distances'''

    '''def process_lidar(self, point_cloud):
        # Example: Convert point cloud to polar coordinates and take min distance per sector
        num_sectors = 8
        sector_width = 360 / num_sectors
        min_distances = np.full(num_sectors, np.inf)

        for point in point_cloud:
            x = point.x  # Extract x coordinate
            y = point.y  # Extract y coordinate
            print("pointcloudx",x)
            angle = np.degrees(np.arctan2(y, x))
            sector = int(angle // sector_width)
            distance = np.sqrt(x ** 2 + y ** 2)
            min_distances[sector] = min(min_distances[sector], distance)

        return min_distances'''

    def process_lidar(self, point_cloud):
        num_sectors = 8
        sector_width = 360 / num_sectors
        min_distances = np.full(num_sectors, 2)

        for point in point_cloud:
            x = point.x
            y = point.y
            print("pointcloudx", x)
            angle = np.degrees(np.arctan2(y, x))
            sector = int((angle + 180) // sector_width) % num_sectors
            print("sector?",sector)
            distance = np.sqrt(x ** 2 + y ** 2)
            min_distances[sector] = min(min_distances[sector], distance)
        print("mindistances",min_distances)
        ''' 
            for i in min_distances:
            if min_distances[i] > 50:
                min_distances[i] = 1.0       #mano
            '''
        return min_distances


env = WebotsEnv()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_webots")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones = env.step(action)
    if dones:
        obs = env.reset()
