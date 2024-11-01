import gymnasium
from numpy.linalg import qr
import manipulator_mujoco  # noqa
import cv2
import sqlite3
import argparse
import numpy as np
import json


def create_db(db):
    db = sqlite3.connect(db)
    db.execute(
        """CREATE TABLE IF NOT EXISTS data (
            seed INTEGER,
            step INTEGER,
            observation BLOB,
            action BLOB,
            info BLOB,
            img BLOB)
            """
    )
    return db


def db_log(db, seed, step, observation, action, info, img):
    observation = json.dumps(observation.tolist())
    action = json.dumps(action)
    info = json.dumps(info)
    db.execute(
        "INSERT INTO data (seed, step, observation, action, info, img) VALUES (?, ?, ?, ?, ?, ?)",
        (seed, step, observation, action, info, img),
    )


def db_start_seed(db, seed):
    db.execute("DELETE FROM data WHERE seed = ?", (seed,))


def sim(seed=42, render_mode="human", db="data.db"):
    db_logging = False
    if render_mode == "rgb_array":
        db = create_db(db)
        db_start_seed(db, seed)
        db_logging = True
    env = gymnasium.make("manipulator_mujoco/UR5eEnv-v0", render_mode=render_mode)

    # Reset the environment with a specific seed for reproducibility
    observation, info = env.reset(seed=seed)

    state = 0
    cnt = 0
    box_target = info["box_pos"]
    state_diff = [0, -0.08, -0.08, 0]
    closing = [False, False, True, True]
    for step in range(10000):
        if cnt > 100:
            cnt = 0
            state += 1
            if state >= len(state_diff):
                break

        target = box_target.copy()
        target[2] += state_diff[state]
        action = target + [250 if closing[state] else 0]
        observation, reward, terminated, truncated, info = env.step(action)
        if db_logging:
            img = env.render()
            db_log(
                db,
                seed,
                step,
                observation,
                action,
                info,
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            )
        x = np.array(observation[:3])
        err = np.linalg.norm(x - target)

        if err < 0.01:
            cnt += 1
        if terminated or truncated:
            break
            # observation, info = env.reset()
    if db_logging:
        db.commit()
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=str, default="data.db")
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    args = parser.parse_args()
    sim(args.seed, args.render_mode, args.db)


if __name__ == "__main__":
    main()
