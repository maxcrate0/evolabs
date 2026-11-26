"""Vectorized 2D navigation demo. Frames reuse the scored trajectories."""

import numpy as np


WIDTH, HEIGHT = 640, 384
STEPS = 90
GOAL = np.array([555, 192], dtype=np.float32)
START = np.array([76, 192], dtype=np.float32)
OBSTACLES = np.array([[303, 153, 45], [405, 253, 42]], dtype=np.float32)
LOWER_LIMIT = np.array([8, 8], dtype=np.float32)
UPPER_LIMIT = np.array([WIDTH - 8, HEIGHT - 8], dtype=np.float32)


def describe():
    return {"inputs": 8, "outputs": 2}


def preview():
    return render_history([START[None, :]], 0, "best")[0]


def evaluate(predict, weights, *, view, seed):
    n = len(weights)
    positions = np.tile(START, (n, 1))
    velocity = np.zeros_like(positions)
    closest = np.full(n, np.linalg.norm(GOAL - START), dtype=np.float32)
    finished = np.zeros(n, dtype=bool)
    bonus = np.zeros(n, dtype=np.float32)
    collisions = np.zeros(n, dtype=np.float32)
    observations = np.empty((n, 8), dtype=np.float32)
    history = [positions.copy()] if view != "none" else None
    for step in range(STEPS):
        observations[:, :2] = (GOAL - positions) / WIDTH
        observations[:, 2:4] = velocity / 8
        observations[:, 4:] = ((OBSTACLES[:, :2] - positions[:, None, :]) / WIDTH).reshape(n, 4)
        velocity *= 0.68
        velocity += np.clip(predict(weights, observations), -1, 1) * 2.6
        velocity[finished] = 0
        proposed = np.clip(positions + velocity, LOWER_LIMIT, UPPER_LIMIT)
        delta = proposed[:, None, :] - OBSTACLES[:, :2]
        hit = np.any(np.sum(delta * delta, axis=2) < (OBSTACLES[:, 2] + 5) ** 2, axis=1)
        collisions += hit
        velocity[hit] = 0
        positions[~hit] = proposed[~hit]
        distances = np.linalg.norm(positions - GOAL, axis=1)
        np.minimum(closest, distances, out=closest)
        reached = (distances <= 16) & ~finished
        bonus[reached] = 65 + (STEPS - step) * 0.7
        finished |= reached
        if history is not None and ((step + 1) % 4 == 0 or step == STEPS - 1):
            history.append(positions.copy())
        if finished.all():
            if history is not None and (step + 1) % 4 and step != STEPS - 1:
                history.append(positions.copy())
            break
    scores = (np.linalg.norm(GOAL - START) - closest) * 0.28 + bonus - collisions * 0.24
    result = {"scores": scores}
    if history is not None:
        best = int(np.argmax(scores))
        result["frames"] = render_history(history, best, view)
    return result


def render_history(history, best, view):
    scenery = [
        {"kind": "circle", "x": float(START[0]), "y": float(START[1]),
         "r": 17, "outline": "#596787"},
        {"kind": "circle", "x": float(GOAL[0]), "y": float(GOAL[1]),
         "r": 16, "fill": "#335d43", "outline": "#a4f58b"},
    ]
    for x, y, radius in OBSTACLES:
        scenery.append({"kind": "circle", "x": float(x), "y": float(y),
                        "r": float(radius), "fill": "#2b354b", "outline": "#52617d"})
    trail = []
    frames = []
    for positions in history:
        x, y = map(float, positions[best])
        trail.extend((x, y))
        shapes = list(scenery)
        if len(trail) >= 4:
            shapes.append({"kind": "line", "points": trail.copy(), "fill": "#528d86", "width": 2})
        if view == "population":
            shapes.extend({"kind": "circle", "id": i, "x": float(px), "y": float(py), "r": 4,
                           "fill": "#7394cf"}
                          for i, (px, py) in enumerate(positions) if i != best)
        shapes.append({"kind": "circle", "id": "best", "x": x, "y": y, "r": 7, "fill": "#71f0c2"})
        frames.append({"width": WIDTH, "height": HEIGHT, "duration_ms": 28,
                       "background": "#141d2d", "shapes": shapes})
    return frames
