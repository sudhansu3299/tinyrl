
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_rewards.csv")

df["rolling_reward"] = (
    df["episode_reward"]
    .rolling(window=20, min_periods=1)
    .mean()
)

plt.figure(figsize=(10, 5))
plt.plot(
    df["environment_steps"],
    df["episode_reward"],
    alpha=0.3,
    label="Episode reward",
)
plt.plot(
    df["environment_steps"],
    df["rolling_reward"],
    label="20-episode moving average",
)
plt.axhline(475, linestyle="--", label="475 reward threshold")
plt.xlabel("Environment steps")
plt.ylabel("Episode reward")
plt.title("PPO CartPole Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)
plt.show()
