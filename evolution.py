"""One population evaluation per generation, independent of the UI."""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class Settings:
    population: int = 96
    generations: int = 100
    elites: int = 4
    parents: int = 24
    mutation_rate: float = 0.20
    mutation_scale: float = 0.18
    seed: int = 7

    def validate(self):
        if self.population < 2 or self.generations < 1:
            raise ValueError("Use ao menos 2 indivíduos e 1 geração.")
        if not 1 <= self.elites <= self.parents <= self.population or self.elites == self.population:
            raise ValueError("Use 1 ≤ elites ≤ pais ≤ população e elites < população.")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("A taxa de mutação deve estar entre 0 e 1.")
        if not np.isfinite(self.mutation_scale) or self.mutation_scale < 0:
            raise ValueError("A magnitude da mutação deve ser finita e não negativa.")
        if self.seed < 0:
            raise ValueError("A semente deve ser não negativa.")


@dataclass(frozen=True)
class Generation:
    number: int
    best: float
    mean: float
    record: float
    seconds: float
    view: str
    frames: list


@dataclass
class TrainingState:
    rows: list = field(default_factory=list)
    population: np.ndarray | None = None
    scores: np.ndarray | None = None
    evolution_rng: dict | None = None
    environment_rng: dict | None = None

    @property
    def completed(self):
        return len(self.rows)

    @property
    def best_index(self):
        return int(np.argmax(self.scores))


def offspring(population, scores, settings, rng):
    ranked = np.argsort(-scores, kind="stable")[:settings.parents]
    child_count = settings.population - settings.elites
    selected = rng.integers(settings.parents, size=child_count)
    next_population = np.empty_like(population)
    next_population[:settings.elites] = population[ranked[:settings.elites]]
    children = next_population[settings.elites:]
    children[:] = population[ranked[selected]]
    rate, scale = settings.mutation_rate, settings.mutation_scale
    if rate and scale:
        if rate == 1:
            children += rng.standard_normal(children.shape, dtype=np.float32) * scale
        else:
            mask = rng.random(children.shape, dtype=np.float32) < rate
            children[mask] += rng.standard_normal(np.count_nonzero(mask), dtype=np.float32) * scale
    return next_population


def train(environment, network, settings, get_view, stop, on_generation, *, state=None):
    settings.validate()
    if state is None:
        state = TrainingState()
    if state.completed:
        if state.population.shape != (settings.population, network.size):
            raise ValueError("A população salva é incompatível com as dimensões da rede atual.")
        rng, env_rng = np.random.default_rng(0), np.random.default_rng(0)
        rng.bit_generator.state = state.evolution_rng
        env_rng.bit_generator.state = state.environment_rng
        record = state.rows[-1][3]
    else:
        evolution_seed, environment_seed = np.random.SeedSequence(settings.seed).spawn(2)
        rng, env_rng = np.random.default_rng(evolution_seed), np.random.default_rng(environment_seed)
        record = float("-inf")

    for number in range(state.completed + 1, settings.generations + 1):
        if stop.is_set():
            break
        if state.completed:
            population = offspring(state.population, state.scores, settings, rng)
        else:
            population = network.population(rng, settings.population)
        view = get_view()
        started = perf_counter()
        seed = int(env_rng.integers(2**32))
        population.setflags(write=False)
        result = environment.evaluate(network, population, view, seed)
        scores = np.array(result["scores"], dtype=np.float64, copy=True)
        if scores.shape != (settings.population,) or not np.isfinite(scores).all():
            raise ValueError(f"O ambiente deve retornar {settings.population} pontuações finitas, na ordem dos pesos.")
        frames = result.get("frames", []) if view != "none" else []
        if not isinstance(frames, list):
            raise ValueError("frames deve ser uma lista de quadros.")
        best = float(scores.max())
        record = max(record, best)
        mean, seconds = float(scores.mean()), perf_counter() - started
        state.population, state.scores = population, scores
        state.evolution_rng = rng.bit_generator.state
        state.environment_rng = env_rng.bit_generator.state
        state.rows.append((number, best, mean, record, seconds, seed))
        on_generation(Generation(number, best, mean, record, seconds, view, frames))
    return state
