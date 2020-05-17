import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def plot(fpath):
    with open(fpath, "rb") as fp:
        scores = pickle.load(fp)

    episodes = np.arange(len(scores)) + 1
    scores = pd.Series(scores)
    avg_scores = scores.rolling(100).mean()

    fig, ax = plt.subplots(1, 1, figsize=[15, 10])
    ax.plot(avg_scores, "-", c="green", linewidth=6)
    ax.plot(scores, "-", c="black", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.grid(which="major")
    ax.legend(["Score of Each Episode", "Moving Average of last 100 Episode", \
     "Criteria"])
    fig.tight_layout()
    fig.savefig("result.png")

if __name__ == "__main__":
    plot("./result.pickle")
