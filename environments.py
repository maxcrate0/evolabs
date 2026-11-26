"""Adapters for in-process Python and persistent executable environments."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import demo
from challenges import ChromeDino, FlappyBird, LanternFlight, MoonRun
from network import load_file


DEFAULT_CHALLENGE = "flappy-bird"
BUILTINS = {DEFAULT_CHALLENGE: FlappyBird(), "chrome-dino": ChromeDino()}
# Saved runs must keep the original rules and network dimensions.
LEGACY_BUILTINS = {"navigation": demo, "lantern-flight": LanternFlight(), "moon-run": MoonRun()}


def dimensions(info):
    inputs, outputs = info["inputs"], info["outputs"]
    if any(type(n) is not int or n < 1 for n in (inputs, outputs)):
        raise ValueError("O ambiente deve informar inputs e outputs como inteiros positivos.")
    return inputs, outputs


class PythonEnvironment:
    def __init__(self, module):
        self.module = module
        self.inputs, self.outputs = dimensions(module.describe())

    def evaluate(self, network, weights, view, seed):
        return self.module.evaluate(network.predict, weights, view=view, seed=seed)


class ExecutableEnvironment:
    def __init__(self, command):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def __enter__(self):
        try:
            self.send({"type": "describe", "version": 1})
            self.inputs, self.outputs = dimensions(self.receive("description"))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process.stdout.close()

    def send(self, message):
        try:
            self.process.stdin.write(json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("O executável fechou o canal de entrada.") from error

    def receive(self, expected=None):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("O executável fechou o canal sem responder.")
        try:
            message = json.loads(line)
        except ValueError as error:
            raise ValueError("O executável deve escrever apenas linhas JSON em stdout.") from error
        if not isinstance(message, dict):
            raise ValueError("Cada linha do protocolo deve ser um objeto JSON.")
        if message.get("type") == "error":
            raise RuntimeError(f"Ambiente: {message.get('message', '')}")
        if expected is not None and message.get("type") != expected:
            raise ValueError(f"Resposta esperada: {expected}.")
        return message

    def evaluate(self, network, weights, view, seed):
        self.send({"type": "evaluate", "view": view, "seed": seed,
                   "network": {"genome_size": network.size, "architecture": network.descriptor},
                   "weights": weights.tolist()})
        while True:
            message = self.receive()
            if message.get("type") == "result":
                return message
            if message.get("type") != "predict":
                raise ValueError("Durante a avaliação, o executável deve enviar predict ou result.")
            rows = message.get("indices")
            selected = weights
            if rows is not None:
                if not isinstance(rows, list) or any(type(i) is not int or not 0 <= i < len(weights) for i in rows):
                    raise ValueError("indices deve conter apenas índices válidos da população.")
                selected = weights[np.asarray(rows, dtype=np.intp)]
            observations = message["observations"]
            if len(selected) == 0 and isinstance(observations, list) and len(observations) == 0:
                observations = np.empty((0, network.inputs), dtype=np.float32)
            actions = network.predict(selected, observations)
            self.send({"type": "prediction", "actions": actions.tolist()})


@contextmanager
def open_environment(path=None, *, external=False):
    path = path or DEFAULT_CHALLENGE
    if path in BUILTINS:
        yield PythonEnvironment(BUILTINS[path])
    elif path in LEGACY_BUILTINS:
        yield PythonEnvironment(LEGACY_BUILTINS[path])
    elif Path(path).suffix.lower() == ".py" and not external:
        yield PythonEnvironment(load_file(path))
    else:
        path = Path(path).resolve(strict=True)
        command = [sys.executable, "-u", str(path)] if path.suffix.lower() == ".py" else [str(path)]
        with ExecutableEnvironment(command) as environment:
            yield environment
