
import gymnasium as gym

#initialization of gym env to an instance
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
# A vector of 4 values: Cart Pos, Cart Vel, Pole Angle, Pole Angular Vel
print(" Starting Observation:", observation) #Current State Representation
print("info", info) #Extra diagnostic info, not usually needed

'''
The observation can be 24 in case of bipedalwalker (24,) and 
action space can be (4,) ; a_t = [a_1,a_2,a_3,a_4]
Each value controls a motor associated with one of the walker's joints. 
The values are bounded between (-1, 1).

observation = [
    hull_angle,
    hull_angular_velocity,
    horizontal_velocity,
    vertical_velocity,
    ...
    left_hip_angle,
    left_knee_angle,
    ...
]
'''
for step in range(3):
    #Sample a random action from the action space
    #Cartpole has 2 discrete actions: 0(left) or 1(right)
    action = env.action_space.sample()
    print("Action Space", env.action_space)
    print("Action:", action)

    observation, reward, terminated, truncated, info = (
        #applies the action, returns new state, reward, internal simulation,
        # & checks if program is terminated or truncated
        env.step(action) # 1 Transition
    )

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    if terminated or truncated:
        observation, info = env.reset() # Starts a new episode

env.close()
