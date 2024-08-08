#@markdown ### **Env Demo**
#@markdown Standard Gym Env (0.21.0 API)
from env.env import PushTImageEnv
import numpy as np
import cv2

# 0. create env object
env = PushTImageEnv()

# 1. seed env for initial state.
# Seed 0-200 are used for the demonstration dataset.
env.seed(1000)

# 2. must reset before use
obs, info = env.reset()

# 3. 2D positional action space [0,512]
action = env.action_space.sample()
for i in range(100):
# 4. Standard gym step method
    #action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    action = info["goal_pose"][0:2]
    print("infogoal_pose:", info["goal_pose"][0:2])
    img = env.render(mode='rgb_array')
    cv2.imshow("d",img)
    cv2.waitKey(0)
# prints and explains each dimension of the observation and action vectors
with np.printoptions(precision=4, suppress=True, threshold=5):
    print("obs['image'].shape:", obs['image'].shape, "float32, [0,1]")
    print("obs['agent_pos'].shape:", obs['agent_pos'].shape, "float32, [0,512]")
    print("action.shape: ", action.shape, "float32, [0,512]")