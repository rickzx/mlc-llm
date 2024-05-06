"""
Implementation for BERT architecture.
"""

import dataclasses
from functools import partial
from typing import Any, Dict, Optional

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BertConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the BERT model."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_act: str
    layer_norm_eps: float
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    head_dim: int = 0
    max_batch_size: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.intermediate_size is None or self.intermediate_size == -1:
            self.intermediate_size = 4 * self.hidden_size
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maxmimum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or `max_sequence_length` is "
                    "provided in `config.json`."
                )
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.head_dim * self.num_attention_heads == self.hidden_size
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %s (%d)",
                bold("prefill_chunk_size"),
                bold("context_window_size"),
                self.context_window_size,
            )
            self.prefill_chunk_size = self.context_window_size
        elif self.prefill_chunk_size > self.context_window_size:
            logger.info(
                "Overriding %s from %d to %d (%s)",
                bold("prefill_chunk_size"),
                self.prefill_chunk_size,
                self.context_window_size,
                bold("context_window_size"),
            )
            self.prefill_chunk_size = self.context_window_size


# pylint: disable=invalid-name,missing-docstring,too-many-locals


class BertSelfAttention(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: BertConfig):
        self.num_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.head_dim = config.head_dim

        self.qkv = nn.Linear(
            in_features=config.hidden_size,
            out_features=3 * self.num_heads * self.head_dim,
            bias=True,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        qkv = self.qkv(hidden_states)
        qkv = op.reshape(qkv, (b, s, 3 * h, d))
        q, k, v = op.split(qkv, 3, axis=2)

        # Attention
        output = op_ext.attention(q, k, v, attention_mask)
        return output


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.silu,
    "swish": nn.silu,
    "gelu_new": partial(nn.gelu, approximate=True),
}


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, dtype="float32")
        self.position_embeddings = nn.Embedding(
            config.context_window_size, config.hidden_size, dtype="float32"
        )
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size, dtype="float32")
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, position_ids: Tensor):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def forward(self, inputs: Tensor, attention_mask: Tensor):
        def _input_positions(inputs: te.Tensor):
            b, s = inputs.shape
            return te.compute((b, s), lambda _, j: j.astype("int32"), name="input_positions")

        input_positions = op.tensor_expr_op(
            _input_positions,
            name_hint="input_positions",
            args=[inputs],
        )

        token_type_ids = op.zeros(inputs.shape, dtype="int32")

        embeddings = self.embeddings(inputs, token_type_ids, input_positions)
        encoder_output = self.encoder(embeddings, attention_mask)
        return encoder_output

    def prefill(self, inputs: Tensor, attention_mask: Tensor):
        def _attention_mask(mask: te.Tensor, zero, batch_size, seq_len):
            return te.compute(
                (batch_size, 1, seq_len, seq_len),
                lambda b, _, i, j: tir.Select(
                    tir.any(mask[b, i] == zero, mask[b, j] == zero),
                    tir.min_value(self.dtype),
                    tir.max_value(self.dtype),
                ),
                name="attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask_2d = op.tensor_expr_op(
            _attention_mask,
            name_hint="attention_mask_prefill",
            args=[attention_mask, tir.IntImm("int32", 0), batch_size, seq_len],
        )
        return self.forward(inputs, attention_mask_2d)
    
    def prefill_binary(self, inputs: Tensor, attention_mask: Tensor):
        def _attention_mask(mask: te.Tensor, zero, batch_size, seq_len):
            return te.compute(
                (batch_size, 1, seq_len, seq_len),
                lambda b, _, i, j: tir.Select(
                    tir.any(mask[b, i] == zero, mask[b, j] == zero),
                    tir.min_value(self.dtype),
                    tir.max_value(self.dtype),
                ),
                name="attention_mask_prefill",
            )

        batch_size, seq_len = inputs.shape
        attention_mask_2d = op.tensor_expr_op(
            _attention_mask,
            name_hint="attention_mask_prefill",
            args=[attention_mask, tir.IntImm("int32", 0), batch_size, seq_len],
        )
        result = self.forward(inputs, attention_mask_2d)  # (batch_size, max_seq_len, hidden_size)

        # Get result[:, 0] > 0
        def _index_and_binary_quant(result: te.Tensor):
            b, s, h = result.shape
            return te.compute(
                (b, h),
                lambda b, h: tir.Select(
                    result[b, 0, h] > 0,
                    1,
                    0
                ),
                name="index_and_binary_quant",
            )

        return op.tensor_expr_op(
            _index_and_binary_quant,
            name_hint="index_and_binary_quant",
            args=[result]
        )


    def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
        return op.softmax(logits / op.reshape(temperature, (temperature.shape[0], 1, 1)), axis=-1)

    def quantize_embedding_binary(self, inputs: Tensor):
        """
        Equivalent to inputs > 0.
        """
        def _quantize_embedding_binary(inputs: te.Tensor):
            batch_size, hidden_size  = inputs.shape
            return te.compute(
                (batch_size, hidden_size),
                lambda b, h: tir.Select(
                    inputs[b, h] > 0,
                    1,
                    0
                ),
                name="quantize_embedding_binary"
            )
        quantized_embedding = op.tensor_expr_op(
            _quantize_embedding_binary,
            name_hint="quantize_embedding_binary",
            args=[inputs]
        )

        return quantized_embedding

    def quantize_embedding_int8(self, inputs: Tensor, range_min: Tensor, range_max: Tensor):
        """
        Quantize the model to int8.

        Parameters
        ----------
        inputs: nn.Tensor
            Input embeddings to be quantized, of shape (batch_size, hidden_size)
        range_min: nn.Tensor
            Min for each hidden_size, received from calibration embeddings of shape (hidden_size,)
        range_max: nn.Tensor
            Max for each hidden_size, received from calibration embeddings of shape (hidden_size,)
        """
        def _quantize_embedding_int8(inputs: te.Tensor, range_min: te.Tensor, range_max: te.Tensor):
            batch_size, hidden_size  = inputs.shape
            return te.compute(
                (batch_size, hidden_size),
                lambda b, h: (inputs[b, h] - range_min[h]) / 
                    ((range_max[h] - range_min[h]) / 255) - 128,
                name="quantize_embedding_int8"
            )
        quantized_embedding = op.tensor_expr_op(
            _quantize_embedding_int8,
            name_hint="quantize_embedding_int8",
            args=[inputs, range_min, range_max]
        )

        return quantized_embedding.astype("int8")

    def get_default_spec(self):
        mod_spec = {
            "prefill": {
                "inputs": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
                "attention_mask": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill_binary": {
                "inputs": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
                "attention_mask": nn.spec.Tensor(["batch_size", "seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "softmax_with_temperature": {
                "logits": nn.spec.Tensor(["batch_size", 1, "vocab_size"], "float32"),
                "temperature": nn.spec.Tensor(["batch_size"], "float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "quantize_embedding_binary": {
                "inputs": nn.spec.Tensor(["batch_size", "hidden_size"], dtype="float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
            "quantize_embedding_int8": {
                "inputs": nn.spec.Tensor(["batch_size", "hidden_size"], dtype="float32"),
                "range_min": nn.spec.Tensor(["hidden_size"], dtype="float32"),
                "range_max": nn.spec.Tensor(["hidden_size"], dtype="float32"),
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            }
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
