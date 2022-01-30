import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

log_path = os.path.join('Training', 'logs')
save_path = os.path.join('Training', "Saved Models", "PPO Models Cartpole")
env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env])
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=40000)
# model.save(save_path)
model = PPO.load(save_path, env=env)
# evaluate_policy(model, env, n_eval_episodes=5, render=True)

def Random_games():
    for episode in range(2):
        obs = env.reset()
        for t in range(500):
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(t, obs, reward, done, info, action)
            if done:
                break

Random_games()