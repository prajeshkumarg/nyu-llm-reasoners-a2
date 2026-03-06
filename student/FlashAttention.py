import torch
import triton
import triton.language as tl


class NaiveFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q: (B, Nq, d)   K: (B, Nk, d)   V: (B, Nk, d)
        B, Nq, d = Q.shape
        _, Nk, _ = K.shape

        Bq = 16  # query tile size
        Bk = 16  # key tile size
        scale = 1.0 / (d ** 0.5)

        Tq = Nq // Bq
        Tk = Nk // Bk

        O = torch.zeros_like(Q)           # (B, Nq, d)
        L = torch.zeros(B, Nq, device=Q.device, dtype=Q.dtype)  # (B, Nq)

        for i in range(Tq):
            q_s = i * Bq
            q_e = q_s + Bq

            Qi = Q[:, q_s:q_e, :]            # (B, Bq, d)
            Oi = torch.zeros_like(Qi)         # (B, Bq, d)
            li = torch.zeros(B, Bq, device=Q.device, dtype=Q.dtype)  # (B, Bq)
            mi = torch.full((B, Bq), -float('inf'), device=Q.device, dtype=Q.dtype)

            for j in range(Tk):
                k_s = j * Bk
                k_e = k_s + Bk

                Kj = K[:, k_s:k_e, :]        # (B, Bk, d)
                Vj = V[:, k_s:k_e, :]        # (B, Bk, d)

                # Scores: (B, Bq, Bk)
                Sij = torch.bmm(Qi, Kj.transpose(1, 2)) * scale

                # New row-wise max: (B, Bq)
                mi_new = torch.max(mi, Sij.max(dim=2).values)

                # Exponentiate with new max: (B, Bq, Bk)
                Pij = torch.exp(Sij - mi_new.unsqueeze(2))

                # Correction factor for old accumulators: (B, Bq)
                alpha = torch.exp(mi - mi_new)

                # Update running sum and output
                li = alpha * li + Pij.sum(dim=2)
                Oi = alpha.unsqueeze(2) * Oi + torch.bmm(Pij, Vj)

                mi = mi_new

            # Normalize
            Oi = Oi / li.unsqueeze(2)
            Li = mi + torch.log(li)

            O[:, q_s:q_e, :] = Oi
            L[:, q_s:q_e] = Li

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
    
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, #these are pointers to Q,K,V
    O_ptr, L_ptr, #ptrs to output
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
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
    
    # Load Q tile -- stays in SRAM for the entire loop
    Qi = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)
    
    # Initialize accumulators
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)  # row-wise max
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)                # row-wise sum
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)              # output accumulator
    
    # Loop over K/V tiles
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for _ in range(num_k_tiles):
        Kj = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
        Vj = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)
        
        # Compute attention scores: (Q_TILE_SIZE, K_TILE_SIZE)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        
        # New row-wise max
        mi_new = tl.maximum(mi, tl.max(Sij, axis=1))
        
        # Correction factor for old accumulators
        alpha = tl.exp(mi - mi_new)
        
        # Exponentiate scores with new max
        Pij = tl.exp(Sij - mi_new[:, None])
        
        # Update running sum and output
        li = alpha * li + tl.sum(Pij, axis=1)
        Oi = alpha[:, None] * Oi + tl.dot(Pij.to(Vj.dtype), Vj)
        
        mi = mi_new
        
        # Advance K and V block pointers to next tile
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    # Normalize output
    Oi = Oi / li[:, None]
    Li = mi + tl.log(li)
    
    # Store output
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, Oi.to(Q_ptr.dtype.element_ty))
    
    # Store log-sum-exp
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
        
        # Allocate output buffers
        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=torch.float32)
        
        # Create the grid: one program per query tile per batch element
        grid = (triton.cdiv(Nq, Q_TILE_SIZE), B)
        
        # Call the kernel
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
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )
        
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


