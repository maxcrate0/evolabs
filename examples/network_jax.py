from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def build(inputs, outputs):
    hidden = 16
    first_end = inputs * hidden
    second_end = first_end + hidden
    third_end = second_end + hidden * outputs
    size = third_end + outputs

    def initialize(rng, count):
        weights = rng.standard_normal((count, size), dtype=np.float32)
        weights[:, :first_end] /= np.sqrt(inputs)
        weights[:, first_end:second_end] = 0
        weights[:, second_end:third_end] /= np.sqrt(hidden)
        weights[:, third_end:] = 0
        return weights

    def individual(genome, observation):
        w1 = genome[:first_end].reshape(inputs, hidden)
        w2 = genome[second_end:third_end].reshape(hidden, outputs)
        state = jnp.tanh(observation @ w1 + genome[first_end:second_end])
        return jnp.tanh(state @ w2 + genome[third_end:])

    return SimpleNamespace(genome_size=size, initialize=initialize,
                           forward=jax.jit(jax.vmap(individual)))
