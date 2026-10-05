import matplotlib
matplotlib.use("Agg")

import io
import base64
import random

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import SGDRegressor

# ---------------------------------------------------------------------------
# Environment: 10 x 10 grid
#   A = agent start   T = target   o = available path   # = wall   D = danger zone
# ---------------------------------------------------------------------------
LAYOUT = [
    "Aoooo#o###",
    "#o##ooDooo",
    "#ooooooo#D",
    "o#oooooooo",
    "oo#oooo#oo",
    "o#oo#D#ooo",
    "ooooo#ooo#",
    "oDoDo#Dooo",
    "oooo#oooDD",
    "ooooD#DooT",
]

ROWS = len(LAYOUT)
COLUMNS = len(LAYOUT[0])

CELL_NAMES = {"A": "Start", "T": "Target", "o": "Path", "#": "Wall", "D": "Danger"}
LEGEND = [
    {"char": "A", "meaning": "Agent start position"},
    {"char": "T", "meaning": "Target (goal)"},
    {"char": "o", "meaning": "Available path"},
    {"char": "#", "meaning": "Wall / obstacle"},
    {"char": "D", "meaning": "Danger zone"},
]


def _find(char):
    return [(r, c) for r in range(ROWS) for c in range(COLUMNS) if LAYOUT[r][c] == char]


START = _find("A")[0]
GOAL = _find("T")[0]

CELL_COUNTS = {k: sum(row.count(k) for row in LAYOUT) for k in "ATo#D"}
assert ROWS == 10 and COLUMNS == 10 and all(len(row) == COLUMNS for row in LAYOUT)
assert CELL_COUNTS == {"A": 1, "T": 1, "o": 68, "#": 20, "D": 10}, CELL_COUNTS

# ---------------------------------------------------------------------------
# Actions and reward system
# ---------------------------------------------------------------------------
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
NUMBER_OF_ACTIONS = len(ACTIONS)
NUMBER_OF_FEATURES = ROWS * COLUMNS * NUMBER_OF_ACTIONS

REWARDS = {
    "normal": -1,    # valid move onto a normal cell: a small cost per step
    "invalid": -5,   # tried to leave the grid: the agent does not move
    "wall": -8,      # bumped into a wall: the agent does not move
    "danger": -25,   # entered a danger zone: allowed, but heavily penalised
    "goal": 100,     # reached the target: episode ends
}
REWARD_TABLE = [
    {"event": "Moving to a valid normal cell (A or o)", "reward": REWARDS["normal"],
     "effect": "Every step costs 1, so shorter routes earn a higher return."},
    {"event": "Attempting an invalid movement (outside the grid)", "reward": REWARDS["invalid"],
     "effect": "The agent stays in place and loses 5; it learns to respect the borders."},
    {"event": "Hitting a wall (#)", "reward": REWARDS["wall"],
     "effect": "The agent stays in place and loses 8; walls are worse than borders."},
    {"event": "Entering a danger zone (D)", "reward": REWARDS["danger"],
     "effect": "The move is allowed, but -25 makes a longer safe detour cheaper than crossing."},
    {"event": "Reaching the target (T)", "reward": REWARDS["goal"],
     "effect": "Large positive reward that ends the episode and anchors all Q-values."},
]

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
EPISODES = 600
MAX_STEPS = 80
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.988
LEARNING_RATE = 0.1
BATCH_SIZE = 8
SEED = 42

PARAMETERS = [
    {"name": "Training episodes", "value": EPISODES,
     "why": "Enough repetitions for the Q-values of a 100-cell grid to propagate back from the target (about 6 s of training)."},
    {"name": "Max steps per episode", "value": MAX_STEPS,
     "why": "Maximum number of allowed steps: ends an episode when the agent wanders; also bounds the final evaluation."},
    {"name": "Discount factor (γ)", "value": GAMMA,
     "why": "Close to 1 so the +100 reward 20 steps away still influences the first moves, "
            "while still preferring shorter paths."},
    {"name": "Initial epsilon (ε)", "value": EPSILON_START,
     "why": "Start fully random: the agent knows nothing and must explore."},
    {"name": "Minimum epsilon", "value": EPSILON_MIN,
     "why": "Keeps 5% exploration so the agent never stops testing alternatives."},
    {"name": "Epsilon decay", "value": EPSILON_DECAY,
     "why": "ε is multiplied by this after every episode, shifting gradually from exploration to exploitation."},
    {"name": "Learning rate (eta0)", "value": LEARNING_RATE,
     "why": "Step size of SGDRegressor.partial_fit; how far each transition moves a Q-value."},
    {"name": "Mini-batch size", "value": BATCH_SIZE,
     "why": "Observed transitions are fed to partial_fit in small groups, which keeps training fast."},
    {"name": "Random seed", "value": SEED,
     "why": "Makes the training reproducible."},
]

# Identity matrix: row i is the one-hot encoding of (state, action) pair i.
_ONE_HOT = np.eye(NUMBER_OF_FEATURES)


def cell_char(position):
    return LAYOUT[position[0]][position[1]]


def step(state, action):
    """Apply one action and describe what happened."""
    row = state[0] + ACTIONS[action][0]
    column = state[1] + ACTIONS[action][1]

    if not (0 <= row < ROWS and 0 <= column < COLUMNS):
        return {"next_state": state, "cell_type": "Outside grid",
                "reward": REWARDS["invalid"], "done": False}

    char = LAYOUT[row][column]
    if char == "#":
        return {"next_state": state, "cell_type": "Wall",
                "reward": REWARDS["wall"], "done": False}

    next_state = (row, column)
    if char == "T":
        return {"next_state": next_state, "cell_type": "Target",
                "reward": REWARDS["goal"], "done": True}
    if char == "D":
        return {"next_state": next_state, "cell_type": "Danger",
                "reward": REWARDS["danger"], "done": False}
    return {"next_state": next_state, "cell_type": CELL_NAMES[char],
            "reward": REWARDS["normal"], "done": False}


def _rows_for(state):
    start = (state[0] * COLUMNS + state[1]) * NUMBER_OF_ACTIONS
    return _ONE_HOT[start:start + NUMBER_OF_ACTIONS]


def predict_q_values(model, state):
    """Q(s, a) for the four actions, estimated by the SGDRegressor."""
    return model.predict(_rows_for(state))


def _greedy_action(q_values, rng=None):
    best = np.flatnonzero(q_values == q_values.max()).tolist()
    return rng.choice(best) if rng else best[0]


def _learning_chart(rewards, successes, epsilons):
    window = 50
    kernel = np.ones(window) / window
    smooth_reward = np.convolve(rewards, kernel, mode="valid")
    smooth_success = np.convolve(successes, kernel, mode="valid") * 100
    x = np.arange(window, len(rewards) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, smooth_reward, color="#0d6efd")
    axes[0].set_title("Reward per Episode (50-episode moving average)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")

    axes[1].plot(x, smooth_success, color="#198754", label="Success rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate (%)", color="#198754")
    ax2 = axes[1].twinx()
    ax2.plot(np.arange(1, len(epsilons) + 1), epsilons, color="#dc3545", linestyle="--")
    ax2.set_ylabel("Epsilon (ε)", color="#dc3545")
    axes[1].set_title("Success Rate and Epsilon Decay")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def train(episodes=EPISODES):
    """Run Q-Learning with an incremental SGDRegressor, then evaluate greedily."""
    rng = random.Random(SEED)
    epsilon = EPSILON_START

    model = SGDRegressor(
        loss="squared_error",
        penalty=None,
        fit_intercept=False,
        learning_rate="constant",
        eta0=LEARNING_RATE,
        random_state=np.random.RandomState(SEED),
    )
    episode_rewards, episode_success, epsilons = [], [], []

    # Skipping scikit-learn's per-call input validation makes the thousands of
    # tiny predict()/partial_fit() calls of reinforcement learning ~3x faster.
    with sklearn.config_context(skip_parameter_validation=True, assume_finite=True):
        # partial_fit must be called once before predict() can be used.
        model.partial_fit(np.zeros((1, NUMBER_OF_FEATURES)), np.array([0.0]))

        for _ in range(episodes):
            state = START
            q_state = predict_q_values(model, state)
            batch_x, batch_y = [], []
            total = 0
            reached = False

            for _ in range(MAX_STEPS):
                # Exploration / exploitation (epsilon-greedy)
                if rng.random() < epsilon:
                    action = rng.randrange(NUMBER_OF_ACTIONS)
                else:
                    action = _greedy_action(q_state, rng)

                outcome = step(state, action)
                next_state, reward, done = outcome["next_state"], outcome["reward"], outcome["done"]

                # Q-value target: r  (+ gamma * max Q(s', a') when not terminal)
                if done:
                    target = float(reward)
                    q_next = None
                else:
                    q_next = predict_q_values(model, next_state)
                    target = reward + GAMMA * float(q_next.max())

                batch_x.append((state[0] * COLUMNS + state[1]) * NUMBER_OF_ACTIONS + action)
                batch_y.append(target)

                state = next_state
                total += reward

                # Incremental learning from a small batch of observed transitions
                if len(batch_x) >= BATCH_SIZE or done:
                    model.partial_fit(_ONE_HOT[batch_x], np.array(batch_y))
                    batch_x, batch_y = [], []
                    q_next = None if done else predict_q_values(model, state)
                q_state = q_next

                if done:
                    reached = True
                    break

            if batch_x:
                model.partial_fit(_ONE_HOT[batch_x], np.array(batch_y))

            episode_rewards.append(total)
            episode_success.append(1 if reached else 0)
            epsilons.append(epsilon)
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # ---- Evaluation without exploration (greedy policy, no model updates) ----
    state = START
    path = [state]
    steps = []
    total_reward = 0
    for number in range(1, MAX_STEPS + 1):
        action = _greedy_action(predict_q_values(model, state))
        outcome = step(state, action)
        steps.append({
            "number": number,
            "state": state,
            "action": ACTION_NAMES[action],
            "next_state": outcome["next_state"],
            "cell_type": outcome["cell_type"],
            "reward": outcome["reward"],
        })
        total_reward += outcome["reward"]
        state = outcome["next_state"]
        path.append(state)
        if outcome["done"]:
            break
    reached_goal = state == GOAL

    # ---- Learned Q-values for every non-wall state ----
    q_table = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            char = LAYOUT[r][c]
            if char in "#T":
                continue
            values = predict_q_values(model, (r, c)).tolist()
            q_table.append({
                "state": (r, c),
                "cell": char,
                "values": values,
                "best": int(np.argmax(values)),
            })

    successes = int(sum(episode_success))
    return {
        "episodes": episodes,
        "successes": successes,
        "success_pct": round(successes / episodes * 100, 1),
        "average_reward": round(float(np.mean(episode_rewards)), 2),
        "average_last_100": round(float(np.mean(episode_rewards[-100:])), 2),
        "final_epsilon": round(epsilon, 4),
        "reached_goal": reached_goal,
        "moves": len(steps),
        "total_reward": total_reward,
        "danger_entries": sum(1 for s in steps if s["cell_type"] == "Danger"),
        "penalties": sum(1 for s in steps if s["cell_type"] in ("Wall", "Outside grid")),
        "path": path,
        "path_cells": {p: i for i, p in enumerate(path)},
        "steps": steps,
        "q_table": q_table,
        "chart": _learning_chart(episode_rewards, episode_success, epsilons),
    }


def epsilon_schedule(checkpoints=(1, 50, 100, 200, 400, EPISODES)):
    """Epsilon used at the start of selected episodes, for the Concepts page."""
    return [
        {"episode": n, "epsilon": round(max(EPSILON_MIN, EPSILON_START * EPSILON_DECAY ** (n - 1)), 3)}
        for n in checkpoints
    ]
