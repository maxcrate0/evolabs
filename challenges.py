"""Original side-scrolling challenges with batched physics and optional frames."""

import numpy as np


WIDTH, HEIGHT = 640, 360
PLAYER_X = 122
RADIUS = 9
TICKS = 460


def rectangle(x, y, w, h, color, edge=""):
    return {"kind": "rect", "x": float(x), "y": float(y), "w": float(w),
            "h": float(h), "fill": color, "outline": edge}


def circle(x, y, r, color, edge=""):
    return {"kind": "circle", "x": float(x), "y": float(y), "r": r, "fill": color, "outline": edge}


class Scroller:
    def describe(self):
        return {"inputs": 5, "outputs": 1}

    def preview(self):
        return self.frame(self.course(0), 0, np.array([self.start]), np.array([True]), 0, "best")

    def evaluate(self, predict, weights, *, view, seed):
        count = len(weights)
        obstacles = self.course(seed)
        ends = obstacles[:, 0] + obstacles[:, 1]
        positions = np.full(count, self.start, dtype=np.float32)
        velocity = np.zeros(count, dtype=np.float32)
        alive = np.ones(count, dtype=bool)
        scores = np.zeros(count, dtype=np.float64)
        observations = np.empty((count, 5), dtype=np.float32)
        snapshots = [(0, positions.copy(), alive.copy())] if view != "none" else None
        for tick in range(1, TICKS + 1):
            progress = tick * self.speed
            index = min(int(np.searchsorted(ends, progress - RADIUS)), len(ends) - 1)
            upcoming = obstacles[index]
            self.observe(observations, positions, velocity, upcoming, progress)
            action = predict(weights, observations)[:, 0]
            self.advance(positions, velocity, action, alive)
            overlap = upcoming[0] <= progress + RADIUS and progress - RADIUS <= ends[index]
            alive &= ~self.collisions(positions, upcoming[2], overlap)
            scores += alive
            running = alive.any()
            if snapshots is not None and (tick % 10 == 0 or not running or tick == TICKS):
                snapshots.append((progress, positions.copy(), alive.copy()))
            if not running:
                break
        result = {"scores": scores}
        if snapshots is not None:
            best = int(np.argmax(scores))
            result["frames"] = [self.frame(obstacles, x, y, active, best, view)
                                for x, y, active in snapshots]
        return result

    def frame(self, obstacles, progress, positions, alive, best, view):
        shapes = self.scenery(progress)
        for index, (distance, width, level) in enumerate(obstacles):
            x = PLAYER_X + distance - progress
            if x + width > 0 and x < WIDTH:
                shapes.extend({**shape, "id": f"obstacle:{index}:{part}"}
                              for part, shape in enumerate(self.obstacle(x, width, level)))
        if view == "population":
            shapes.extend({**circle(PLAYER_X, y, 5, self.crowd), "id": f"agent:{i}"}
                          for i, y in enumerate(positions) if i != best and alive[i])
        if alive[best] or view == "best":
            shapes.extend({**shape, "id": f"best:{part}"}
                          for part, shape in enumerate(self.agent(positions[best], bool(alive[best]))))
        return {"width": WIDTH, "height": HEIGHT, "duration_ms": 30,
                "background": self.background, "shapes": shapes}


class LanternFlight(Scroller):
    start = 170
    speed = 3.6
    gap = 136
    background = "#221b32"
    crowd = "#bc91d1"

    def course(self, seed):
        rng = np.random.default_rng(seed)
        centers = np.clip(171 + np.cumsum(rng.integers(-52, 53, 18)), 100, 258)
        return np.column_stack((230 + np.arange(18) * 176, np.full(18, 42), centers))

    def observe(self, obs, y, vy, upcoming, progress):
        distance, width, center = upcoming
        obs[:, 0] = (center - y) / HEIGHT
        obs[:, 1] = vy / 9
        obs[:, 2] = (distance - progress) / 200
        obs[:, 3] = y / HEIGHT
        obs[:, 4] = (distance + width - progress) / 200

    def advance(self, y, vy, action, alive):
        # A lift impulse is available on the downward phase.
        lift = (action > 0.1) & (vy > 0)
        vy[:] = np.minimum(np.where(lift, -5.1, vy + 0.36), 8)
        y += vy * alive

    def collisions(self, y, center, overlap):
        return (y < RADIUS) | (y > HEIGHT - RADIUS) | (overlap & (np.abs(y - center) > self.gap / 2 - RADIUS))

    def scenery(self, progress):
        return [{**circle((i * 97 - progress * 0.15) % WIDTH, 21 + (i * 53) % 309, 2, "#796289"),
                 "id": f"star:{i}"}
                for i in range(14)]

    def obstacle(self, x, width, center):
        top, bottom = center - self.gap / 2, center + self.gap / 2
        return [
            rectangle(x, 0, width, top, "#463456", "#88679e"),
            rectangle(x, bottom, width, HEIGHT - bottom, "#463456", "#88679e"),
            circle(x + width / 2, top - 9, 3, "#e9b4fa"),
            circle(x + width / 2, bottom + 9, 3, "#e9b4fa"),
        ]

    def agent(self, y, alive):
        return [
            circle(PLAYER_X, y, 12, "#6c4839", "#f7ca81" if alive else "#a56e6a"),
            circle(PLAYER_X, y, 6, "#ffcf81" if alive else "#b47873"),
        ]


class MoonRun(Scroller):
    ground = 300
    start = ground - RADIUS
    speed = 4.6
    background = "#17283a"
    crowd = "#7baccb"

    def course(self, seed):
        rng = np.random.default_rng(seed)
        distances = 205 + np.r_[0, np.cumsum(rng.integers(158, 231, 17))]
        return np.column_stack((distances, rng.integers(16, 44, 18), rng.integers(18, 59, 18)))

    def observe(self, obs, y, vy, upcoming, progress):
        distance, width, height = upcoming
        obs[:, 0] = (distance - progress) / 225
        obs[:, 1] = width / 50
        obs[:, 2] = height / 60
        obs[:, 3] = (self.start - y) / 100
        obs[:, 4] = vy / 12

    def advance(self, y, vy, action, alive):
        ground = y >= self.start
        vy[:] = np.where(ground, np.where(action > 0.25, -10.4, 0), vy + 0.63)
        y[:] = np.minimum(self.start, y + vy * alive)

    def collisions(self, y, height, overlap):
        return overlap & (y + RADIUS > self.ground - height)

    def scenery(self, progress):
        return [
            circle(510, 82, 38, "#304b63"),
            rectangle(0, self.ground, WIDTH, HEIGHT - self.ground, "#243d52"),
            *[{**rectangle((i * 76 - progress * 0.8) % WIDTH, self.ground + 12 + (i % 2) * 17,
                           20, 2, "#3c5f74"), "id": f"track:{i}"} for i in range(10)],
        ]

    def obstacle(self, x, width, height):
        return [rectangle(x, self.ground - height, width, height, "#476a80", "#93c2d1"),
                rectangle(x + 2, self.ground - height + 3, width - 4, 3, "#bef1d9")]

    def agent(self, y, alive):
        return [
            circle(PLAYER_X, y, RADIUS, "#a1efdf" if alive else "#91869c", "#c9fff5"),
            circle(PLAYER_X + 3, y - 2, 3, "#3c7084"),
        ]
