
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="LunarLander-v3")
args = parser.parse_args()

path = f"runs/{args.env}/training_rewards.csv"
df = pd.read_csv(path)

df["rolling_reward"] = (
    df["episode_reward"].rolling(20, min_periods=1).mean()
)

plt.figure(figsize=(10, 5))
plt.plot(
    df["environment_steps"], df["episode_reward"],
    alpha=0.3, label="Episode reward"
)
plt.plot(
    df["environment_steps"], df["rolling_reward"],
    label="20-episode average"
)
plt.axhline(200, linestyle="--", label="Reward 200 reference")
plt.xlabel("Environment steps")
plt.ylabel("Episode reward")
plt.title(f"PPO learning curve — {args.env}")
plt.grid(True)
plt.legend()
plt.tight_layout()

output = f"runs/{args.env}/learning_curve.png"
plt.savefig(output, dpi=150)
plt.show()

print(f"Saved plot to {output}")
