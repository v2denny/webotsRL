import random


def get_positions(mode):
    # Positions (init): [(final1),(final2)...]
    # Easy Positions: robot is placed closer to the target
    easypositions = {
        (-0.562, 0.187): [(-1.313, 0.187)], #Map1
        (-1.313, 0.559): [(-1.313, 0.187)], #Map1
        (-0.066, 0.187): [(-1.313, 0.187)], #Map1
        (0.769, 0.313): [(1.311, 0.187)], #Map2
        (0.744, 1.368): [(1.314, 1.311)], #Map2
        (1.185, -1.438): [(1.187, -1.059)], #Map4
        (1.190, -0.566): [(1.187, -1.059)], #Map4
    }

    # Hard Positions: robot and target have a lot of obstacles in between
    hardpositions = {
        (-1.316, 1.312): [(-1.313, 0.187)], #Map1
        (-0.813, 1.310): [(-1.313, 0.187)], #Map1
        (-0.186, 0.439): [(-1.313, 0.187)], #Map1
        (0.186, 0.309): [(1.311, 0.187), (1.314, 1.311)], #Map2
        (0.062, 1.435): [(1.311, 0.187), (1.314, 1.311)], #Map2
        (1.310, 0.681): [(1.311, 0.187), (1.314, 1.311)], #Map2
        (-1.380, -0.747): [(-1.063, -0.313), (-1.063, -1.188)], #Map3
        (-1.380, -1.185): [(-1.063, -0.313), (-1.063, -1.188)], #Map3
        (-1.380, -0.315): [(-1.063, -0.313), (-1.063, -1.188)], #Map3
        (-0.063, -1.187): [(-1.063, -0.313), (-1.063, -1.188)], #Map3
        (-0.253, -0.751): [(-1.063, -0.313), (-1.063, -1.188)], #Map3
        (0.126, -1.369): [(1.187, -1.059)], #Map4
        (0.063, -0.063): [(1.187, -1.059)], #Map4
    }

    # Select random start and target positions
    if mode == "easy":
        initial_position = random.choice(list(easypositions.keys()))
        #final_position = random.choice(easypositions[initial_position])
        final_position = easypositions[initial_position]
    
    elif mode == "hard":
        initial_position = random.choice(list(hardpositions.keys()))
        #final_position = random.choice(hardpositions[initial_position])
        final_position = hardpositions[initial_position]


    return initial_position, final_position
