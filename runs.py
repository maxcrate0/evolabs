"""Portable archive for training metrics and the latest evaluated population."""

import csv
import io
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from environments import BUILTINS, DEFAULT_CHALLENGE, LEGACY_BUILTINS, open_environment
from evolution import Settings, TrainingState
from network import build_network


DEFAULT_RUNS = Path(__file__).resolve().parent / "treinos"
COLUMNS = ("generation", "best", "mean", "record", "evaluation_seconds", "environment_seed")


@dataclass(frozen=True)
class Experiment:
    environment: str
    architecture: dict | str
    python: bool = False
    external: bool = False

    @classmethod
    def create(cls, environment, architecture, *, python=False, external=False):
        if environment and environment not in BUILTINS and environment not in LEGACY_BUILTINS:
            environment = str(Path(environment).resolve())
        if python:
            architecture = str(Path(architecture).resolve())
        elif isinstance(architecture, str):
            architecture = json.loads(architecture)
        return cls(environment or DEFAULT_CHALLENGE, architecture, python, external)

    @contextmanager
    def open(self):
        with open_environment(self.environment, external=self.external) as environment:
            network = build_network(self.architecture, environment.inputs, environment.outputs, python=self.python)
            yield environment, network


@dataclass(frozen=True)
class SavedRun:
    path: Path
    experiment: Experiment
    settings: Settings
    state: TrainingState
    outcome: str

    def continuation(self, rounds):
        return (replace(self.settings, generations=self.state.completed + rounds),
                replace(self.state, rows=self.state.rows.copy()))

    def replay(self):
        with self.experiment.open() as (environment, network):
            if self.state.population.shape[1] != network.size:
                raise ValueError("A arquitetura atual é incompatível com os pesos salvos.")
            best = self.state.best_index
            weights = self.state.population[best:best + 1]
            result = environment.evaluate(network, weights, "best", self.state.rows[-1][-1])
            frames = result.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError("O ambiente não retornou quadros para esta rede.")
            return frames


def save_run(experiment, settings, state, outcome, directory=DEFAULT_RUNS):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now(timezone.utc)
    stamp = saved_at.strftime("%Y-%m-%d_%H-%M-%S")
    path = directory / f"treino_{stamp}_{uuid4().hex[:8]}.evolab"
    manifest = {
        "version": 1, "saved_at": saved_at.isoformat(), "outcome": outcome,
        "experiment": asdict(experiment), "settings": asdict(settings),
        "evolution_rng": state.evolution_rng, "environment_rng": state.environment_rng,
    }
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(COLUMNS)
    writer.writerows(state.rows)
    best_weights = state.population[state.best_index]
    with NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with ZipFile(staging, "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("training.json", json.dumps(manifest, ensure_ascii=False, allow_nan=False))
            archive.writestr("history.csv", table.getvalue().encode("utf-8"))
            for name, array in (("best.npy", best_weights), ("population.npy", state.population),
                                ("scores.npy", state.scores)):
                with archive.open(name, "w") as file:
                    np.save(file, array, allow_pickle=False)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
    return SavedRun(path, experiment, settings, state, outcome)


def load_run(path):
    path = Path(path).resolve()
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("training.json"))
        if manifest["version"] != 1:
            raise ValueError("Versão do arquivo de treino incompatível.")
        experiment = Experiment(**manifest["experiment"])
        settings = Settings(**manifest["settings"])
        settings.validate()
        table = csv.reader(io.StringIO(archive.read("history.csv").decode("utf-8")))
        if tuple(next(table, ())) != COLUMNS:
            raise ValueError("O histórico do treino está inválido.")
        rows = [(int(number), float(best), float(mean), float(record), float(seconds), int(seed))
                for number, best, mean, record, seconds, seed in table]
        with archive.open("population.npy") as file:
            population = np.load(file, allow_pickle=False)
        with archive.open("scores.npy") as file:
            scores = np.load(file, allow_pickle=False)
    if (not rows or len(rows) > settings.generations
            or any(row[0] != number or not 0 <= row[-1] < 2**32
                   or not all(np.isfinite(row[1:5])) for number, row in enumerate(rows, 1))
            or population.ndim != 2 or population.shape[0] != settings.population
            or population.dtype != np.float32 or not np.isfinite(population).all()
            or scores.shape != (settings.population,) or not np.isfinite(scores).all()):
        raise ValueError("O arquivo contém uma população ou histórico inválido.")
    population.setflags(write=False)
    state = TrainingState(rows, population, scores, manifest["evolution_rng"], manifest["environment_rng"])
    return SavedRun(path, experiment, settings, state, manifest["outcome"])
