"""Side-scrolling games with batched physics and optional frames."""

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
    acceleration = 0
    frame_interval = 10
    half_width = RADIUS

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
            progress = tick * (self.speed + (tick - 1) * self.acceleration / 2)
            index = min(int(np.searchsorted(ends, progress - self.half_width)), len(ends) - 1)
            upcoming = obstacles[index]
            self.observe(observations, positions, velocity, upcoming, progress)
            action = predict(weights, observations)[:, 0]
            self.advance(positions, velocity, action, alive)
            overlap = upcoming[0] <= progress + self.half_width and progress - self.half_width <= ends[index]
            alive &= ~self.collisions(positions, upcoming[2], overlap)
            scores += alive
            running = alive.any()
            if snapshots is not None and (tick % self.frame_interval == 0 or not running or tick == TICKS):
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


class Flight(Scroller):
    def observe(self, obs, y, vy, upcoming, progress):
        distance, width, center = upcoming
        obs[:, 0] = (center - y) / HEIGHT
        obs[:, 1] = vy / 9
        obs[:, 2] = (distance - progress) / 200
        obs[:, 3] = y / HEIGHT
        obs[:, 4] = (distance + width - progress) / 200

    def collisions(self, y, center, overlap):
        return (y < RADIUS) | (y > self.ground - RADIUS) | (overlap & (np.abs(y - center) > self.gap / 2 - RADIUS))


class LanternFlight(Flight):
    start = 170
    speed = 3.6
    gap = 136
    ground = HEIGHT
    background = "#221b32"
    crowd = "#bc91d1"

    def course(self, seed):
        rng = np.random.default_rng(seed)
        centers = np.clip(171 + np.cumsum(rng.integers(-52, 53, 18)), 100, 258)
        return np.column_stack((230 + np.arange(18) * 176, np.full(18, 42), centers))

    def advance(self, y, vy, action, alive):
        # A lift impulse is available on the downward phase.
        lift = (action > 0.1) & (vy > 0)
        vy[:] = np.minimum(np.where(lift, -5.1, vy + 0.36), 8)
        y += vy * alive

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


class Runner(Scroller):
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


class MoonRun(Runner):
    ground = 300
    start = ground - RADIUS
    speed = 4.6
    background = "#17283a"
    crowd = "#7baccb"

    def course(self, seed):
        rng = np.random.default_rng(seed)
        distances = 205 + np.r_[0, np.cumsum(rng.integers(158, 231, 17))]
        return np.column_stack((distances, rng.integers(16, 44, 18), rng.integers(18, 59, 18)))

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


class FlappyBird(Flight):
    start = 160
    speed = 3.6
    gap = 116
    ground = 320
    half_width = 12
    frame_interval = 6
    background = "#70c5ce"
    crowd = "#fff2a3"

    def course(self, seed):
        rng = np.random.default_rng(seed)
        centers = rng.integers(106, 225, 18)
        return np.column_stack((175 + np.arange(18) * 182, np.full(18, 52), centers))

    def advance(self, y, vy, action, alive):
        vy[:] = np.minimum(np.where(action > 0.1, -5.1, vy + 0.36), 8)
        y += vy * alive

    def scenery(self, progress):
        clouds = []
        for i in range(5):
            x = (i * 143 - progress * 0.18) % (WIDTH + 100) - 50
            y = 72 + (i % 3) * 54
            clouds.extend([
                {**circle(x, y, 22, "#d6f2ef"), "id": f"cloud:{i}:left"},
                {**circle(x + 25, y + 6, 17, "#d6f2ef"), "id": f"cloud:{i}:right"},
            ])
        return [
            *clouds,
            rectangle(0, self.ground, WIDTH, HEIGHT - self.ground, "#ded895"),
            rectangle(0, self.ground, WIDTH, 8, "#8dcc47", "#5b9031"),
            *[{**rectangle((i * 36 - progress) % (WIDTH + 36) - 18, self.ground + 9,
                            18, 3, "#bbb46d"), "id": f"grass:{i}"} for i in range(19)],
        ]

    def obstacle(self, x, width, center):
        top, bottom = center - self.gap / 2, center + self.gap / 2
        return [
            rectangle(x, 0, width, top, "#74bf33", "#365c21"),
            rectangle(x + 7, 0, 6, top - 14, "#a6dc53"),
            rectangle(x, top - 14, width, 14, "#8bd13c", "#365c21"),
            rectangle(x, bottom, width, self.ground - bottom, "#74bf33", "#365c21"),
            rectangle(x + 7, bottom + 14, 6, self.ground - bottom - 14, "#a6dc53"),
            rectangle(x, bottom, width, 14, "#8bd13c", "#365c21"),
        ]

    def agent(self, y, alive):
        return [
            circle(PLAYER_X, y, RADIUS, "#ffdb48" if alive else "#c2ad65", "#674524"),
            circle(PLAYER_X - 5, y + 2, 4, "#fff1a8", "#af7930"),
            rectangle(PLAYER_X + 5, y + 1, 7, 4, "#f58b3d", "#674524"),
            circle(PLAYER_X + 4, y - 4, 4, "#ffffff"),
            circle(PLAYER_X + 6, y - 4, 1.5, "#392e29"),
        ]


class ChromeDino(Runner):
    ground = 275
    start = ground - RADIUS
    speed = 4.7
    acceleration = 0.006
    half_width = 12
    frame_interval = 6
    background = "#f7f7f7"
    crowd = "#9b9b9b"
    ink = "#535353"

    def course(self, seed):
        rng = np.random.default_rng(seed)
        distances = 194 + np.r_[0, np.cumsum(rng.integers(177, 235, 17))]
        return np.column_stack((distances, rng.choice([16, 24, 36], 18), rng.integers(22, 43, 18)))

    def scenery(self, progress):
        return [
            rectangle(0, self.ground - 1, WIDTH, 1, self.ink),
            *[{**rectangle((i * 67 - progress) % (WIDTH + 20) - 10, self.ground + 5 + (i % 3) * 4,
                            5 + (i % 3) * 4, 1, "#a0a0a0"), "id": f"track:{i}"} for i in range(10)],
        ]

    def obstacle(self, x, width, height):
        top = self.ground - height
        return [
            rectangle(x + width * 0.37, top, width * 0.26, height, self.ink),
            rectangle(x, top + height * 0.30, width * 0.23, height * 0.43, self.ink),
            rectangle(x, top + height * 0.60, width * 0.50, height * 0.13, self.ink),
            rectangle(x + width * 0.77, top + height * 0.18, width * 0.23, height * 0.39, self.ink),
            rectangle(x + width * 0.50, top + height * 0.44, width * 0.50, height * 0.13, self.ink),
        ]

    def agent(self, y, alive):
        color = self.ink if alive else "#999999"
        return [
            rectangle(PLAYER_X - 12, y - 3, 4, 5, color),
            rectangle(PLAYER_X - 9, y, 14, 5, color),
            rectangle(PLAYER_X - 4, y - 5, 10, 10, color),
            rectangle(PLAYER_X + 1, y - 16, 11, 10, color),
            rectangle(PLAYER_X, y - 8, 7, 9, color),
            rectangle(PLAYER_X + 5, y - 3, 5, 2, color),
            rectangle(PLAYER_X - 6, y + 4, 4, 5, color),
            rectangle(PLAYER_X + 1, y + 4, 4, 5, color),
            rectangle(PLAYER_X + 4, y - 14, 2, 2, self.background),
        ]
