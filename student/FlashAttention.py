import math
import torch
import triton
import triton.language as tl


class NaiveFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape

        Bq = 16
        Bk = 16
        scale = 1.0 / (d ** 0.5)

        Tq = math.ceil(Nq / Bq)
        Tk = math.ceil(Nk / Bk)

        O = torch.zeros_like(Q)
        L = torch.zeros(B, Nq, device=Q.device, dtype=Q.dtype)

        for i in range(Tq):
            q_s = i * Bq
            q_e = min(q_s + Bq, Nq)
            tile_q = q_e - q_s

            Qi = Q[:, q_s:q_e, :]
            Oi = torch.zeros_like(Qi)
            li = torch.zeros(B, tile_q, device=Q.device, dtype=Q.dtype)
            mi = torch.full((B, tile_q), -float('inf'), device=Q.device, dtype=Q.dtype)

            for j in range(Tk):
                k_s = j * Bk
                k_e = min(k_s + Bk, Nk)

                Kj = K[:, k_s:k_e, :]
                Vj = V[:, k_s:k_e, :]

                Sij = torch.bmm(Qi, Kj.transpose(1, 2)) * scale

                if is_causal:
                    q_indices = torch.arange(q_s, q_e, device=Q.device).unsqueeze(1)
                    k_indices = torch.arange(k_s, k_e, device=Q.device).unsqueeze(0)
                    causal_mask = k_indices > q_indices
                    Sij = Sij.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

                mi_new = torch.max(mi, Sij.max(dim=2).values)
                Pij = torch.exp(Sij - mi_new.unsqueeze(2))
                alpha = torch.exp(mi - mi_new)

                li = alpha * li + Pij.sum(dim=2)
                Oi = alpha.unsqueeze(2) * Oi + torch.bmm(Pij, Vj)

                mi = mi_new

            Oi = Oi / li.unsqueeze(2)
            Li = mi + torch.log(li)

            O[:, q_s:q_e, :] = Oi
            L[:, q_s:q_e] = Li

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        dQ, dK, dV = _flash_bwd(Q, K, V, O, dO, L, scale, is_causal)
        return dQ, dK, dV, None


@torch.compile
def _flash_bwd(Q, K, V, O, dO, L, scale, is_causal):
    scale = 1.0 / (Q.shape[-1] ** 0.5)

    D = (dO * O).sum(dim=-1)
    S = torch.bmm(Q, K.transpose(1, 2)) * scale

    if is_causal:
        q_idx = torch.arange(S.shape[1], device=S.device).unsqueeze(1)
        k_idx = torch.arange(S.shape[2], device=S.device).unsqueeze(0)
        S = S.masked_fill(k_idx > q_idx, float('-inf'))

    P = torch.exp(S - L.unsqueeze(-1))

    dV = torch.bmm(P.transpose(1, 2), dO)
    dP = torch.bmm(dO, V.transpose(1, 2))
    dS = P * (dP - D.unsqueeze(-1)) * scale
    dQ = torch.bmm(dS, K)
    dK = torch.bmm(dS.transpose(1, 2), Q)

    return dQ, dK, dV


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Qi = tl.load(Q_block_ptr)

    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(num_k_tiles):
        Kj = tl.load(K_block_ptr)
        Vj = tl.load(V_block_ptr)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        if IS_CAUSAL:
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = k_offsets[None, :] > q_offsets[:, None]
            Sij = tl.where(causal_mask, float('-inf'), Sij)

        mi_new = tl.maximum(mi, tl.max(Sij, axis=1))
        alpha = tl.exp(mi - mi_new)
        Pij = tl.exp(Sij - mi_new[:, None])

        li = alpha * li + tl.sum(Pij, axis=1)
        Oi = alpha[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj)

        mi = mi_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, Oi.to(Q_ptr.dtype.element_ty))

    L_ptrs = L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    tl.store(L_ptrs, Li)


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape

        scale = 1.0 / (d ** 0.5)

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(Nq, Q_TILE_SIZE), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            IS_CAUSAL=is_causal,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = ctx.scale
        dQ, dK, dV = _flash_bwd(Q, K, V, O, dO, L, scale, is_causal)
        return dQ, dK, dV, None
