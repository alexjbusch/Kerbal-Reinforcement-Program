import gym
import KerbalLanderEnvironment

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


ENV_NAME = 'KerbalLander-v0'
ksp_env = gym.make(ENV_NAME)
check_env(ksp_env, warn=True)

ksp_vec_env = make_vec_env('KerbalLander-v0', n_envs=1, env_kwargs=dict(grid_size=10))
model = A2C("MlpPolicy", ksp_env, verbose=1).learn(5000)

obs = ksp_vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = ksp_vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    ksp_vec_env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break
