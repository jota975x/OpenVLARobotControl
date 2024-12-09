import torch
import pybullet as p
import numpy as np
import random
import math
from scripts.lidar import lidar_sim

WHEEL_SEPARATION = 0.2
WHEEL_RADIUS = 0.05
random.seed(42)

def diff_drive_control(linear_vel=0.0, angular_vel=0.0):
    v_right = linear_vel + (angular_vel * WHEEL_SEPARATION / 2)
    v_left = linear_vel - (angular_vel * WHEEL_SEPARATION / 2)
    
    return [- v_right / WHEEL_RADIUS, v_left / WHEEL_RADIUS]

def get_state(robot_id, goal):

    # get robot position and velocities
    robot_pos, robot_orientation = p.getBasePositionAndOrientation(robot_id)
    robot_orientation = p.getEulerFromQuaternion(robot_orientation)
    robot_coords = np.array(robot_pos[0:2])
    robot_angle = np.array([robot_orientation[2]])

    robot_lin_vel, robot_w_vel = p.getBaseVelocity(robot_id)

    # transform linear and angular velocities to robot local frame
    rot_matrix = np.array([
        [np.cos(robot_orientation[2]), -np.sin(robot_orientation[2]), 0],
        [np.sin(robot_orientation[2]), np.cos(robot_orientation[2]), 0],
        [0, 0, 1]
    ])
    local_v = np.dot(rot_matrix.T, robot_lin_vel)
    local_w = np.dot(rot_matrix.T, robot_w_vel)

    lin_vel = np.array([local_v[0]])
    ang_vel = np.array([local_w[2]])

    # obtain relative position to the goal
    diff_x = goal[0] - robot_pos[0]
    diff_y = goal[1] - robot_pos[1]
    dist_to_goal = np.array([np.sqrt(diff_x**2 + diff_y**2)])
    angle_to_goal = np.array([np.arctan2(diff_y, diff_x) - robot_orientation[2]])
    
    lidar_data = lidar_sim(robot_id, 2)

    state = np.concatenate([robot_coords, robot_angle, lin_vel, ang_vel, dist_to_goal, angle_to_goal, lidar_data])

    return state

def get_spawn(radius=20.0):
    
    spawn_x = random.uniform(-radius/2, radius/2)
    spawn_y = random.uniform(-radius/2, radius/2)
    spawn_theta = random.uniform(0, 2*math.pi)
    
    return [spawn_x, spawn_y, spawn_theta]

def get_goal(spawn, raidus=10.0):
    angle = random.uniform(0, 2* math.pi)
    dist = random.uniform(0, raidus)
    
    goal_x = spawn[0] + dist * math.cos(angle)
    goal_y = spawn[1] + dist * math.sin(angle)
    
    return [goal_x, goal_y]
    

def compute_reward_done(state, action, next_state):
    k1, k2, k3, k4, k5, k6 = 10.0, 0.05, 0.1, 0.001, 0.01, 0.5
    done = False

    # goal proximity reward
    dist_reward = k1 * (1 / next_state[5])
    if next_state[5] < 0.05:
        dist_reward += 500.0
        done = True

    # angular deviation
    ang_reward = -k2 * np.abs(next_state[6])

    # smooth motion penalty
    smoothness_penalty = -k3 * np.abs(action[1])

    # energy penalty
    energy_penalty = -k4 * (action[0]**2 + action[1]**2)

    # time penalty
    time_penalty = -k5
    
    # obstacle penalty
    obstacle_penalty = 0.0
    min_lidar = np.min(next_state[7:])
    if min_lidar < 0.25:
        obstacle_penalty += -k6 * (1 - min_lidar)
    if min_lidar < 0.05:
        obstacle_penalty = -1000
        done = True

    total_reward = dist_reward + ang_reward + smoothness_penalty + energy_penalty + time_penalty + obstacle_penalty

    return total_reward, done

def add_random_blocks(num_blocks, x_range, y_range):
    size = 0.25
    
    col_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size])
    vis_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[random.random(), random.random(), random.random(), 1])

    blocks = []
    for _ in range(num_blocks):
        x, y, z = random.uniform(*x_range), random.uniform(*y_range), size
        block = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape, basePosition=[x, y, z])
        blocks.append(block)
        
    # place beach ball
    ball_path = 'OpenVLA_PybulletEnv/obj_files/beach_ball.obj'
    visual_ball_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=ball_path,
                                         meshScale=[0.01, 0.01, 0.01])
    collision_ball_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=ball_path,
                                               meshScale=[0.1, 0.1, 0.1])
    ball_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=collision_ball_id, baseVisualShapeIndex=visual_ball_id,
                                basePosition=[5, 5, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    
    # place cat ball
    cat_path = 'OpenVLA_PybulletEnv/obj_files/cat.obj'
    visual_cat_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=cat_path,
                                         meshScale=[0.01, 0.01, 0.01],
                                         rgbaColor=[0.2, 0.2, 0.8, 1.0])
    collision_cat_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=cat_path,
                                               meshScale=[0.1, 0.1, 0.1])
    cat_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision_cat_id, baseVisualShapeIndex=visual_cat_id,
                                basePosition=[2, 8, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    
    # place dog ball
    dog_path = 'OpenVLA_PybulletEnv/obj_files/dog.obj'
    visual_dog_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=dog_path,
                                         meshScale=[0.01, 0.01, 0.01])
    collision_dog_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=dog_path,
                                               meshScale=[0.1, 0.1, 0.1])
    dog_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision_dog_id, baseVisualShapeIndex=visual_dog_id,
                                basePosition=[4, 2, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    
    # place cow ball
    cow_path = 'OpenVLA_PybulletEnv/obj_files/cow.obj'
    visual_cow_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=cow_path,
                                         meshScale=[0.01, 0.01, 0.01])
    collision_cow_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=cow_path,
                                               meshScale=[0.1, 0.1, 0.1])
    cow_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision_cow_id, baseVisualShapeIndex=visual_cow_id,
                                basePosition=[-4, -5, 0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    
    # place pineapple ball
    pine_path = 'OpenVLA_PybulletEnv/obj_files/pineapple.obj'
    visual_pine_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=pine_path,
                                         meshScale=[0.01, 0.01, 0.01])
    collision_pine_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=pine_path,
                                               meshScale=[0.1, 0.1, 0.1])
    pine_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision_pine_id, baseVisualShapeIndex=visual_pine_id,
                                basePosition=[-5, -7, 0.1], baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
    