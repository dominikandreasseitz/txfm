from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Module
from jax import Array


class Block(Module):
    embedding_size: int
    head_size: int
    drop_prob: float = 0.1

    @nn.compact
    def __call__(self, tokens: Array) -> Array:
        T = len(tokens)
        qr = nn.Dense(self.embedding_size, self.head_size, name="query")(tokens)
        ky = nn.Dense(self.embedding_size, self.head_size, name="key")(tokens)
        val = nn.Dense(self.embedding_size, self.head_size, name="value")(tokens)
        att = qr.T @ ky
        att.at[jnp.tril_indices(n=att.shape[0], m=att.shape[1])].set(0.0)
        att = nn.activation.softmax(att)
        att = nn.Dropout(self.drop_prob, deterministic=True, name="drop")(att)
        out = att @ val.T
        return nn.Dense(T, self.embedding_size, name="proj")(out)


@dataclass
class Embedding:
    n_vocab: int
    embedding_size: int

    def __post_init__(self) -> None:
        self.weights = jax.random.uniform(
            jax.random.key(0), shape=(self.n_vocab, self.embedding_size), minval=0.0, maxval=1.0
        )

    def __call__(self, idx: int) -> Array:
        return self.weights[idx, :]


@dataclass
class Tokenizer:
    def encode(self, string: str) -> Array:
        return jnp.array(list(map(self.encd, string)))

    def decode(self, indices: Array) -> str:
        return "".join(list(map(self.decd, indices)))

    def vanilla_tokenize(self, input: str) -> None:
        self.vocab = list(set(input))
        self.vocab_size = len(self.vocab)
        self.encode_dict = {v: k for k, v in enumerate(self.vocab)}
        self.decode_dict = {k: v for k, v in enumerate(self.vocab)}
        self.encd = lambda k: self.encode_dict[k]
        self.decd = lambda k: self.decode_dict[k.item()]

    def bytepair_tokenize(self, input: str, n_merges: int) -> None:
        self.encode_dict = {x: ord(x) for x in input}
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}
        tokens = list(self.decode_dict.keys())

        def max_pair(tokens: list[int]) -> tuple[int, int]:
            occ_dict = {}
            for u0, u1 in zip(tokens[0:-1], tokens[1:]):
                if (u0, u1) not in occ_dict:
                    occ_dict[(u0, u1)] = 1
                else:
                    occ_dict[(u0, u1)] += 1
            return [k for k, v in occ_dict.items() if v == max(occ_dict.values())][0]

        new_token_id = 255 + 1  # Starting from 255 due to utf8 encoding max

        def replace(
            tokens: list[int], pair_to_replace: tuple[int, int], new_token: int
        ) -> list[int]:
            for i in range(len(tokens) - 2):
                if (tokens[i], tokens[i + 1]) == pair_to_replace:
                    tokens = tokens[:i] + [new_token] + tokens[i + 2 :]
            return tokens

        for _ in range(n_merges):
            pair = max_pair(tokens)
            self.encode_dict[pair] = new_token_id
            self.decode_dict[new_token_id] = pair
            tokens = replace(tokens, pair, new_token_id)
            new_token_id += 1


class Transformer(Module):
    tknz: Tokenizer
    embd: Embedding
    blocks: list[Block]
    head_size: int
    vocab_size: int

    def forward(self, input: str) -> Array:
        x = self.tknz.encode(input)
        x = self.embd(x)
        for b in self.blocks:
            x = b(x)
        x = nn.Dense(self.head_size, self.vocab_size, name="ff")(x)
        return x


if __name__ == "__main__":
    input = "hello my name is something"
    tknzr = Tokenizer()
    tknzr.vanilla_tokenize(input)
    test_enc = tknzr.encode(input)
    # tknzr.bytepair_tokenize(input, 2)
    embed = Embedding(n_vocab=tknzr.vocab_size, embedding_size=100)
    test_array = jax.vmap(embed, in_axes=(0,))(test_enc)
    block = Block(embed.embedding_size, head_size=100)
    block_params = block.init(jax.random.key(0), test_array)
    block = block.bind(block_params)
    x = block(test_array)
