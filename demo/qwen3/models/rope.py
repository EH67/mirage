import triton
import triton.language as tl
import torch
from typing import TypedDict


@triton.jit
def rotary_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    seq_len,
    num_kv_heads,
    half_dim,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    cos_stride_s,
    cos_stride_d,
    sin_stride_s,
    sin_stride_d,
    out_q_stride_b,
    out_q_stride_s,
    out_q_stride_h,
    out_q_stride_d,
    out_k_stride_b,
    out_k_stride_s,
    out_k_stride_h,
    out_k_stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    if pid_s * BLOCK_M >= seq_len:
        return

    rm = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    s_mask = rm < seq_len

    q_base = pid_b * q_stride_b + pid_h * q_stride_h
    out_q_base = pid_b * out_q_stride_b + pid_h * out_q_stride_h

    rk_half = tl.arange(0, BLOCK_K // 2)
    rk_half_mask = rk_half < half_dim

    q_offset = q_base + (
        rm[:, None] * q_stride_s + rk_half[None, :] * q_stride_d
    )  # half

    cos_offset = rm[:, None] * cos_stride_s + rk_half[None, :] * cos_stride_d
    sin_offset = rm[:, None] * sin_stride_s + rk_half[None, :] * sin_stride_d

    cos = tl.load(
        cos_ptr + cos_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=1.0
    )
    sin = tl.load(
        sin_ptr + sin_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0
    )

    q_first = tl.load(
        q_ptr + q_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0
    )
    q_second = tl.load(
        q_ptr + q_offset + half_dim * q_stride_d,
        mask=s_mask[:, None] & rk_half_mask[None, :],
        other=0.0,
    )

    q_out_first = q_first * cos - q_second * sin
    q_out_second = q_second * cos + q_first * sin

    out_q_offset = out_q_base + (
        rm[:, None] * out_q_stride_s + rk_half[None, :] * out_q_stride_d
    )

    tl.store(
        out_q_ptr + out_q_offset,
        q_out_first,
        mask=s_mask[:, None] & rk_half_mask[None, :],
    )
    tl.store(
        out_q_ptr + out_q_offset + half_dim * out_q_stride_d,
        q_out_second,
        mask=s_mask[:, None] & rk_half_mask[None, :],
    )

    if pid_h < num_kv_heads:
        k_base = pid_b * k_stride_b + pid_h * k_stride_h
        k_offset = k_base + (rm[:, None] * k_stride_s + rk_half[None, :] * k_stride_d)

        k_first = tl.load(
            k_ptr + k_offset, mask=s_mask[:, None] & rk_half_mask[None, :], other=0.0
        )
        k_second = tl.load(
            k_ptr + k_offset + half_dim * k_stride_d,
            mask=s_mask[:, None] & rk_half_mask[None, :],
            other=0.0,
        )
        k_out_first = k_first * cos - k_second * sin
        k_out_second = k_second * cos + k_first * sin
        out_k_base = pid_b * out_k_stride_b + pid_h * out_k_stride_h

        out_k_offset = out_k_base + (
            rm[:, None] * out_k_stride_s + rk_half[None, :] * out_k_stride_d
        )
        tl.store(
            out_k_ptr + out_k_offset,
            k_out_first,
            mask=s_mask[:, None] & rk_half_mask[None, :],
        )
        tl.store(
            out_k_ptr + out_k_offset + half_dim * out_k_stride_d,
            k_out_second,
            mask=s_mask[:, None] & rk_half_mask[None, :],
        )


def apply_rotary_pos_emb_triton(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Triton implementation of apply_rotary_pos_emb with 3D grid.

    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim]
        cos/sin: Tensors from rotary embedding [1, seq_len_ro, head_dim]
        position_ids: Optional tensor of position indices [batch, seq_len]
        unsqueeze_dim: Dimension to unsqueeze the cosine/sine tensors

    Returns:
        Tuple of rotated query and key tensors
    """

    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    half_dim = head_dim // 2

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    q_stride = q.stride()
    k_stride = k.stride()
    cos_stride = cos.stride()
    sin_stride = sin.stride()
    q_out_stride = q_out.stride()
    k_out_stride = k_out.stride()

    BLOCK_K = (
        32
        if head_dim <= 32
        else (64 if head_dim <= 64 else (128 if head_dim <= 128 else 256))
    )
    BLOCK_M = 8 if head_dim <= 128 else 4

    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size, num_heads)
    with torch.cuda.device(q.device):
        rotary_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            seq_len,
            num_kv_heads,
            half_dim,
            q_stride[0],
            q_stride[1],
            q_stride[2],
            q_stride[3],
            k_stride[0],
            k_stride[1],
            k_stride[2],
            k_stride[3],
            cos_stride[1],
            cos_stride[2],
            sin_stride[1],
            sin_stride[2],
            q_out_stride[0],
            q_out_stride[1],
            q_out_stride[2],
            q_out_stride[3],
            k_out_stride[0],
            k_out_stride[1],
            k_out_stride[2],
            k_out_stride[3],
            BLOCK_M,
            BLOCK_K,
        )
    return q_out, k_out

# Copied from transformers/src/transformers/modeling_rope_utils.py
class RopeParameters(TypedDict, total=False):
    """
    Args:
        rope_theta (`float`):
            The base period of the RoPE embeddings.
        rope_type (`str`, *optional*, defaults to "default"):
            The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
            'llama3'], with 'default' being the original RoPE implementation.
        partial_rotary_factor (`float`, *optional*):
            The percentage of the query and key head embedding on which RoPE will be applied.
        factor (`float`, *optional*):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        original_max_position_embeddings (`int`, *optional*):
            Used with 'yarn', 'longrope' and 'llama3'. The original max position embeddings used during
            pretraining.
        attention_factor (`float`, *optional*):
            Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        short_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to short contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        long_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to long contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        low_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
        high_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
    """

    rope_theta: float
    rope_type: str | None
    partial_rotary_factor: float | None
    factor: float | None
    original_max_position_embeddings: int | None
    attention_factor: float | None
    beta_fast: float | None
    beta_slow: float | None
    short_factor: list[float] | None
    long_factor: list[float] | None
    low_freq_factor: float | None
    high_freq_factor: float | None
