"""Architectures share a flat, float32 genome across the population."""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np


TEMPLATES = {
    "Compacta · 16": {"hidden": [16], "activation": "tanh", "output": "tanh"},
    "Multicamadas · 32 × 16": {"hidden": [32, 16], "activation": "relu", "output": "tanh"},
    "Linear": {"hidden": [], "activation": "tanh", "output": "tanh"},
}

ACTIVATIONS = {
    "linear": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0),
    "sigmoid": lambda x: 1 / (1 + np.exp(-np.clip(x, -80, 80))),
    "softmax": lambda x: _softmax(x),
}


def _softmax(x):
    u = np.exp(x - x.max(axis=1, keepdims=True))
    return u / u.sum(axis=1, keepdims=True)


def load_file(path):
    """Load a local Python plugin with normal imports from its own directory."""
    path = Path(path).resolve()
    name = f"_evolab_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Arquivo Python inválido: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
        sys.modules.pop(name, None)
    return module


@dataclass(frozen=True)
class Network:
    inputs: int
    outputs: int
    size: int
    initialize: object
    forward: object
    descriptor: dict | None = None

    def population(self, rng, count):
        weights = np.asarray(self.initialize(rng, count), dtype=np.float32)
        if weights.shape != (count, self.size) or not np.isfinite(weights).all():
            raise ValueError(f"initialize deve retornar pesos finitos de forma ({count}, {self.size}).")
        return np.array(weights, dtype=np.float32, order="C", copy=True)

    def predict(self, weights, observations):
        weights = np.asarray(weights, dtype=np.float32)
        observations = np.asarray(observations, dtype=np.float32)
        if weights.ndim != 2 or weights.shape[1] != self.size:
            raise ValueError(f"Os pesos precisam ter forma (N, {self.size}).")
        if observations.shape != (len(weights), self.inputs):
            raise ValueError(f"As observações precisam ter forma ({len(weights)}, {self.inputs}).")
        actions = np.asarray(self.forward(weights, observations), dtype=np.float32)
        if actions.shape != (len(weights), self.outputs):
            raise ValueError(f"forward deve retornar forma ({len(weights)}, {self.outputs}).")
        return actions


def dense_network(config, inputs, outputs):
    keys = {"hidden", "activation", "output"}
    if not isinstance(config, dict) or config.keys() != keys:
        raise ValueError("A arquitetura deve definir hidden, activation e output.")
    hidden = config["hidden"]
    if not isinstance(hidden, list) or any(type(n) is not int or n < 1 for n in hidden):
        raise ValueError("hidden deve ser uma lista de inteiros positivos.")
    activation, output = config["activation"], config["output"]
    if any(not isinstance(name, str) or name not in ACTIVATIONS for name in (activation, output)):
        raise ValueError("Ativações disponíveis: " + ", ".join(ACTIVATIONS))

    dimensions = [inputs, *hidden, outputs]
    layers = []
    cursor = 0
    for index, (before, after) in enumerate(zip(dimensions, dimensions[1:])):
        matrix_end = cursor + before * after
        bias_end = matrix_end + after
        function = ACTIVATIONS[output if index == len(dimensions) - 2 else activation]
        layers.append((before, after, cursor, matrix_end, bias_end, function))
        cursor = bias_end

    def initialize(rng, count):
        genomes = np.zeros((count, cursor), dtype=np.float32)
        for before, after, start, end, _, _ in layers:
            genomes[:, start:end] = rng.standard_normal(
                (count, end - start), dtype=np.float32
            ) * np.sqrt(2 / (before + after))
        return genomes

    def forward(weights, observations):
        values = observations
        for before, after, start, end, bias_end, function in layers:
            matrices = weights[:, start:end].reshape(len(weights), before, after)
            values = function(np.einsum("ni,nij->nj", values, matrices) + weights[:, end:bias_end])
        return values

    return Network(inputs, outputs, cursor, initialize, forward,
                   {"type": "dense", "inputs": inputs, "outputs": outputs, **config})


def build_network(source, inputs, outputs, *, python=False):
    if not python:
        config = json.loads(source) if isinstance(source, str) else source
        return dense_network(config, inputs, outputs)
    module = load_file(source)
    model = module.build(inputs, outputs)
    size = model.genome_size
    if type(size) is not int or size < 1:
        raise ValueError("genome_size deve ser um inteiro positivo.")
    if not callable(model.initialize) or not callable(model.forward):
        raise ValueError("O modelo deve implementar initialize e forward.")
    return Network(inputs, outputs, size, model.initialize, model.forward)
