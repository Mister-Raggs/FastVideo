# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn

import fastvideo.envs as envs
from fastvideo.attention import (DistributedAttention, DistributedAttention_VSA,
                                 LocalAttention)
from fastvideo.configs.models.dits import WanVideoConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard)
from fastvideo.layers.layernorm import (FP32LayerNorm, LayerNormScaleShift,
                                        RMSNorm, ScaleResidual,
                                        ScaleResidualLayerNormScaleShift)
from fastvideo.layers.linear import ReplicatedLinear
# from torch.nn import RMSNorm
# TODO: RMSNorm ....
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.layers.visual_embedding import (ModulateProjection, PatchEmbed,
                                               TimestepEmbedder)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum, current_platform

from fastvideo.distributed.parallel_state import get_sp_group, get_sp_world_size
from fastvideo.forward_context import get_forward_context

logger = init_logger(__name__)


class WanImageEmbedding(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self,
                encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        dtype = encoder_hidden_states_image.dtype
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states).to(dtype)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
        self.time_modulation = ModulateProjection(dim,
                                                  factor=6,
                                                  act_layer="silu")
        self.text_embedder = MLP(text_embed_dim,
                                 dim,
                                 dim,
                                 bias=True,
                                 act_type="gelu_pytorch_tanh") if text_embed_dim > 0 else None

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        if self.text_embedder is not None:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        else:
            encoder_hidden_states = torch.zeros((timestep.shape[0], 0, temb.shape[-1]), device=temb.device, dtype=temb.dtype)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 parallel_attention=False) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention

        # layers
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Scaled dot product attention
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
                v = self.to_v(context)[0].view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
            v = self.to_v(context)[0].view(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v) if k.size(1) > 0 else torch.zeros_like(q)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: tuple[AttentionBackendEnum, ...]
        | None = None
    ) -> None:
        super().__init__(dim, num_heads, window_size, qk_norm, eps,
                         supported_attention_backends)

        self.add_k_proj = ReplicatedLinear(dim, dim)
        self.add_v_proj = ReplicatedLinear(dim, dim)
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_added_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(
            b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)
        # compute attention
        x = self.attn(q, k, v) if k.size(1) > 0 else torch.zeros_like(q)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTransformerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1")
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32

        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            e = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
                6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output, _ = self.attn1(
            query,
            key,
            value,
            original_seq_len,
            freqs_cis=freqs_cis,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformerBlock_VSA(nn.Module):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = ""):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_gate_compress = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1")
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1))

        attn_output, _ = self.attn1(
            query,
            key,
            value,
            original_seq_len,
            freqs_cis=freqs_cis,
            gate_compress=gate_compress,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig(
    )._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str,
                                                               Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        assert config.num_attention_heads % get_sp_world_size() == 0, f"The number of attention heads ({config.num_attention_heads}) must be divisible by the sequence parallel size ({get_sp_world_size()})"

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        attn_backend = envs.FASTVIDEO_ATTENTION_BACKEND
        transformer_block = WanTransformerBlock_VSA if attn_backend == "VIDEO_SPARSE_ATTN" else WanTransformerBlock
        self.blocks = nn.ModuleList([
            transformer_block(inner_dim,
                              config.ffn_dim,
                              config.num_attention_heads,
                              config.qk_norm,
                              config.cross_attn_norm,
                              config.eps,
                              config.added_kv_proj_dim,
                              self._supported_attention_backends,
                              prefix=f"{config.prefix}.blocks.{i}")
            for i in range(config.num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32,
                                            compute_dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # DBCache (CacheDiT-style step caching). Config is pushed from
        # FastVideoArgs by the DenoisingStage before each generation; these are
        # safe defaults so an un-configured eager call is bit-identical to main.
        self.use_dbcache = False
        self.dbcache_fn_compute_blocks = 8
        self.dbcache_bn_compute_blocks = 0
        self.dbcache_residual_threshold = 0.08
        self.dbcache_max_warmup_steps = 8
        # Per-generation cache state, keyed by within-step call index (slot 0 =
        # first forward of a step, slot 1 = second, ... — separates the cond and
        # uncond CFG passes, which are sequential forwards sharing a step index).
        self._dbcache_state: dict[int, dict[str, torch.Tensor | None]] = {}
        self._dbcache_last_step_idx: int | None = None
        self._dbcache_call_in_step = 0

        self.__post_init__()

    def reset_dbcache_state(self) -> None:
        """Clear all cached residuals. Call once at the start of a generation."""
        self._dbcache_state = {}
        self._dbcache_last_step_idx = None
        self._dbcache_call_in_step = 0

    def _forward_blocks_dbcache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
    ) -> torch.Tensor:
        """DBCache block loop: always run the first ``Fn`` and last ``Bn``
        blocks; skip the middle span when the Fn-block residual barely changed
        from the previous step (per CFG slot) and reuse a cached residual.

        The skip decision is reduced across the sequence-parallel group so every
        rank runs the identical set of blocks (otherwise the SP collectives
        inside the blocks would desync)."""
        ctx = get_forward_context()
        step_idx = int(ctx.current_timestep) if ctx is not None else 0

        # Detect a new denoise step (cond + uncond share the same step index) and
        # reset the within-step call counter so cond/uncond map to stable slots.
        if step_idx != self._dbcache_last_step_idx:
            self._dbcache_last_step_idx = step_idx
            self._dbcache_call_in_step = 0
        else:
            self._dbcache_call_in_step += 1
        slot = self._dbcache_call_in_step
        st = self._dbcache_state.setdefault(slot, {
            "prev_r_fn": None,
            "mid_residual": None
        })

        n = len(self.blocks)
        fn = max(1, min(self.dbcache_fn_compute_blocks, n))
        bn = max(0, min(self.dbcache_bn_compute_blocks, n - fn))
        fn_blocks = self.blocks[:fn]
        mid_blocks = self.blocks[fn:n - bn] if bn > 0 else self.blocks[fn:]
        bn_blocks = self.blocks[n - bn:] if bn > 0 else []

        # 1. Fn blocks always run; r_fn is the residual they contribute.
        h_in = hidden_states
        h = h_in
        for block in fn_blocks:
            h = block(h, encoder_hidden_states, timestep_proj, freqs_cis,
                      original_seq_len)
        r_fn = h - h_in
        h_fn = h

        # 2. Skip decision (lockstep across SP ranks).
        in_warmup = step_idx < self.dbcache_max_warmup_steps
        do_skip = False
        if (not in_warmup and st["prev_r_fn"] is not None
                and st["mid_residual"] is not None):
            prev = st["prev_r_fn"]
            num = (r_fn - prev).abs().sum().float()
            den = prev.abs().sum().float()
            if get_sp_world_size() > 1:
                packed = torch.stack([num, den])
                packed = get_sp_group().all_reduce(packed)
                num, den = packed[0], packed[1]
            diff = num / (den + 1e-8)
            do_skip = bool(diff < self.dbcache_residual_threshold)

        # 3. Middle blocks: skip + reuse cached residual, or run + cache it.
        if do_skip:
            h_mid = h_fn + st["mid_residual"]
        else:
            h = h_fn
            for block in mid_blocks:
                h = block(h, encoder_hidden_states, timestep_proj, freqs_cis,
                          original_seq_len)
            h_mid = h
            st["mid_residual"] = h_mid - h_fn
        st["prev_r_fn"] = r_fn

        # 4. Bn blocks always run.
        h = h_mid
        for block in bn_blocks:
            h = block(h, encoder_hidden_states, timestep_proj, freqs_cis,
                      original_seq_len)
        return h

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height,
             post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000)
        freqs_cis = (freqs_cos.to(hidden_states.device).float(),
                     freqs_sin.to(hidden_states.device).float())

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Shard with padding support - returns (sharded_tensor, original_seq_len)
        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
        
        current_seq_len = hidden_states.shape[1]
        sp_world_size = get_sp_world_size()
        padded_seq_len = current_seq_len * sp_world_size
        

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len)
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states_image

        if current_platform.is_mps() or current_platform.is_npu():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)
        else:
            encoder_hidden_states = encoder_hidden_states # cast to orig_dtype for MPS & NPU

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis, original_seq_len)
        elif self.use_dbcache:
            hidden_states = self._forward_blocks_dbcache(
                hidden_states, encoder_hidden_states, timestep_proj, freqs_cis,
                original_seq_len)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                      timestep_proj, freqs_cis, original_seq_len)
        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            

        hidden_states = self.norm_out(hidden_states, shift, scale)

        # Gather and unpad in one operation
        hidden_states = sequence_model_parallel_all_gather_with_unpad(
            hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, p_t, p_h, p_w,
                                              -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

# Entry point for model registry
EntryClass = WanTransformer3DModel
