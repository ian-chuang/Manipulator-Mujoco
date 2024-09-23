import re
import gymnasium
from numpy.linalg import qr
import manipulator_mujoco  # noqa
import cv2
import sqlite3
import argparse
import numpy as np
import json


def db_actions(db, seed):
    cursor = db.execute("SELECT action FROM data WHERE seed = ? ORDER BY step", (seed,))
    return cursor.fetchall()


def replay(seed=42, db="data.db"):
    db = sqlite3.connect(db)
    env = gymnasium.make("manipulator_mujoco/UR5eEnv-v0", render_mode="human")

    env.reset(seed=seed)

    for action in db_actions(db, seed):
        action = json.loads(action[0])
        env.step(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=str, default="data.db")
    args = parser.parse_args()
    replay(args.seed, args.db)


if __name__ == "__main__":
    main()
