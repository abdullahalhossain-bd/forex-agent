"""Test script: load the trained PPO model and run a single predict."""
from pathlib import Path
import sys
import numpy as np

from config import PROJECT_ROOT
from ml.rl_agent import RLAgent, _DEFAULT_RL_POLICY_PATH


def main():
    agent = RLAgent()
    model_path = _DEFAULT_RL_POLICY_PATH
    print(f"Model path: {model_path}")
    exists = model_path.exists()
    print(f"Model file exists: {exists}")
    ok = agent.load_model(model_path)
    print(f"load_model returned: {ok}")
    if ok:
        # create a dummy state matching observation dim (24,)
        state = np.zeros((24,), dtype=float)
        action = agent.predict(state)
        print("Predict result:", action.to_dict())
    else:
        print("Model not loaded. Directory listing:")
        try:
            for p in model_path.parent.iterdir():
                print(" -", p.name)
        except Exception as e:
            print("Failed to list directory:", e)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Test script failed:", e)
        sys.exit(2)
