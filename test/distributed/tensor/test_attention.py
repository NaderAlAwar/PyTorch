# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import unittest

from functools import lru_cache
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._ops import TorchDispatchMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh, distribute_tensor, DTensor, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental._attention import (
    _AttentionContextParallel,
    _CausalBehavior,
    _cp_options,
    _is_causal_behavior,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Replicate
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import _mask_mod_signature, create_block_mask
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    ModelArgs,
    Transformer,
    with_comms,
)

c10d_functional = torch.ops.c10d_functional
backends = []
if PLATFORM_SUPPORTS_FLASH_ATTENTION:
    backends.append(SDPBackend.FLASH_ATTENTION)
if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
    backends.append(SDPBackend.EFFICIENT_ATTENTION)


rotater_enum_to_str = {
    _RotateMethod.ALL_GATHER: "allgather",
    _RotateMethod.ALL_TO_ALL: "alltoall",
}  # mapping from _RotateMethod enum to string


class RingAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False

    @skip_if_lt_x_gpu(2)
    @skipIfRocm  # Missing _c10d_functional_autograd::all_to_all_single
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    @with_comms
    def test_ring_attention_sdpa(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "compiled": [True, False],
                "backend": backends,
                "load_balance": [True, False],
                "rotater": [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER],
                "test_forward_only": [True, False],
            },
            self._test_ring_attention_sdpa,
        )

    def _test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
        test_forward_only: bool,
    ) -> None:
        def fn_eval(fn, *args, **kwargs):
            if test_forward_only:
                with torch.no_grad():
                    return fn(*args, **kwargs)
            else:
                out = fn(*args, **kwargs)
                out.sum().backward()
                return out

        if load_balance and not is_causal:
            return

        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        dtype = torch.bfloat16
        bs = 8
        query_tokens = 64
        context_tokens = 64
        dim = 32
        nheads = 8
        torch.manual_seed(10)
        dtype = (
            torch.bfloat16 if backend == SDPBackend.FLASH_ATTENTION else torch.float32
        )

        _cp_options.enable_load_balance = load_balance

        q = torch.rand(
            (bs, nheads, self.world_size * query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )

        # Ensure all ranks have the same initialization data.
        with torch.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

        with sdpa_kernel(backend):
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q = q.detach().clone()
        cp_k = k.detach().clone()
        cp_v = v.detach().clone()
        # Theoretically, context_parallel() should not be used to shard
        # parameters because when require_grad is True, resize_ is not
        # allowed. But requires_grad of cp_q, cp_k, and cp_v are False
        # now. So we can just use context_parallel() to shard q, k, v.
        # In reality, context_paralle() should be used to shard the input.
        with context_parallel(
            device_mesh, buffers=(cp_q, cp_k, cp_v), buffer_seq_dims=(2, 2, 2)
        ):
            cp_q.requires_grad = True
            cp_k.requires_grad = True
            cp_v.requires_grad = True
            with CommDebugMode() as comm_mode:
                with sdpa_kernel(backend):
                    if compiled:
                        fn = torch.compile(
                            F.scaled_dot_product_attention,
                            fullgraph=True,
                            backend="aot_eager",
                        )
                    else:
                        fn = F.scaled_dot_product_attention

                    cp_out = fn_eval(fn, cp_q, cp_k, cp_v, is_causal=is_causal)

                    if not compiled and rotater == _RotateMethod.ALL_TO_ALL:
                        # Compiler and CommDebugMode do not work well together.
                        expect_all2all_count = (
                            self.world_size - 1
                            if test_forward_only
                            else self.world_size * 3 - 2
                        )
                        self.assertDictEqual(
                            comm_mode.get_comm_counts(),
                            {c10d_functional.all_to_all_single: expect_all2all_count},
                        )

            # Due to numerical error, we need to choose different atol for different
            # attention kernels
            (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])
            atol = (
                1e-08
                if backend == SDPBackend.EFFICIENT_ATTENTION
                else 1e-3 * self.world_size
            )
            self.assertTrue(torch.allclose(out, cp_out, atol=atol))

            if not test_forward_only:
                cp_dq, cp_dk, cp_dv = context_parallel_unshard(
                    device_mesh,
                    [cp_q.grad, cp_k.grad, cp_v.grad],
                    [2, 2, 2],
                )
                atol = (
                    2e-06
                    if backend == SDPBackend.EFFICIENT_ATTENTION
                    else 8e-3 * self.world_size
                )
                self.assertTrue(torch.allclose(q.grad, cp_dq, atol=atol))
                self.assertTrue(torch.allclose(k.grad, cp_dk, atol=atol))
                self.assertTrue(torch.allclose(v.grad, cp_dv, atol=atol))

                cp_q.grad = None
                cp_k.grad = None
                cp_v.grad = None

            cp_q.requires_grad = False
            cp_k.requires_grad = False
            cp_v.requires_grad = False

    def test_is_causal_behavior(self) -> None:
        _cp_options.enable_load_balance = False
        self.assertEqual(
            _is_causal_behavior(rank=0, world_size=4, i=0, is_causal=False),
            _CausalBehavior.NOT_IS_CAUSAL,
        )

        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.SKIP],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )

        _cp_options.enable_load_balance = True
        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_ring_attention_native_transformer(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "rotater": [_RotateMethod.ALL_GATHER, _RotateMethod.ALL_TO_ALL],
            },
            self._test_ring_attention_native_transformer,
        )

    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def _test_ring_attention_native_transformer(
        self, is_causal: bool, rotater: _RotateMethod
    ) -> None:
        _cp_options.enable_load_balance = is_causal
        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        dtype = torch.bfloat16
        bs = 8
        ntokens = 8
        dim = 32
        nheads = 8
        num_layers = 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nheads,
            dim_feedforward=dim,
            batch_first=True,
        ).to(dtype)
        encoder_layer = parallelize_module(
            module=encoder_layer,
            device_mesh=device_mesh,
            parallelize_plan={
                "self_attn": _AttentionContextParallel(),
            },
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        model = model.to(self.device_type).to(dtype)

        mask = (
            nn.Transformer.generate_square_subsequent_mask(
                ntokens, device=self.device_type, dtype=dtype
            )
            if is_causal
            else None
        )
        seq = torch.rand((bs, ntokens, dim), device=self.device_type, dtype=dtype)

        with CommDebugMode() as comm_mode:
            out = model(seq, mask=mask, is_causal=is_causal)

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size - 1)
                    * num_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_gather_into_tensor: num_layers,
                },
            )

        with CommDebugMode() as comm_mode:
            out.sum().backward()

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size * 2 - 1)
                    * num_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_gather_into_tensor: num_layers,
                    c10d_functional.all_to_all_single: self.world_size * num_layers,
                },
            )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    @sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    def test_ring_attention_custom_transformer(self) -> None:
        self.run_subtests(
            {"rotater": [_RotateMethod.ALL_GATHER, _RotateMethod.ALL_TO_ALL]},
            self._test_ring_attention_custom_transformer,
        )

    def _test_ring_attention_custom_transformer(self, rotater: _RotateMethod) -> None:
        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, self.world_size),
        )
        dtype = torch.bfloat16
        bs = 2
        args = ModelArgs()

        model = Transformer(args).to(dtype).to(self.device_type)

        model = parallelize_module(
            module=model,
            device_mesh=device_mesh,
            parallelize_plan={
                f"layers.{i}.attention": _AttentionContextParallel()
                for i in range(args.n_layers)
            },
        )

        seq = torch.randint(
            args.vocab_size, (bs, args.max_seq_len), device=self.device_type
        )

        with CommDebugMode() as comm_mode:
            out = model(seq)

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size - 1)
                    * args.n_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {c10d_functional.all_gather_into_tensor: args.n_layers},
            )

        with CommDebugMode() as comm_mode:
            out.sum().backward()

        if rotater == _RotateMethod.ALL_TO_ALL:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_to_all_single: (self.world_size * 2 - 1)
                    * args.n_layers,
                },
            )
        else:
            self.assertDictEqual(
                comm_mode.get_comm_counts(),
                {
                    c10d_functional.all_gather_into_tensor: args.n_layers,
                    c10d_functional.all_to_all_single: self.world_size * args.n_layers,
                },
            )


class RingFlexAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # return torch.cuda.device_count()
        return 2

    @with_comms
    def test_ring_flex_attention(self) -> None:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        from torch.nn.attention.flex_attention import flex_attention

        # Compile the flex_attention function
        flex_attention = torch.compile(flex_attention, dynamic=False)

        torch.manual_seed(10)
        dtype = torch.float32
        bs = 8
        query_tokens = 512 * self.world_size
        context_tokens = 512 * self.world_size
        dim = 32
        nheads = 8

        q = torch.rand(
            (bs, nheads, query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.rand(
            (bs, nheads, context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.rand(
            (bs, nheads, context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )

        block_mask = create_block_mask(
            causal_mask,
            B=bs,
            H=nheads,
            Q_LEN=query_tokens,
            KV_LEN=context_tokens,
            device=self.device_type,
        )

        out = flex_attention(q, k, v, block_mask=block_mask)

        expect_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        torch.testing.assert_close(out, expect_out, atol=1e-1, rtol=1e-2)

        # test flex attention on DTensor
        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )

        q_dist = distribute_tensor(q, device_mesh, [Shard(-2)])
        k_dist = distribute_tensor(k, device_mesh, [Shard(-2)])
        v_dist = distribute_tensor(v, device_mesh, [Shard(-2)])
        assert isinstance(q_dist, DTensor)
        with CPMode():
            out_dt = flex_attention(q_dist, k_dist, v_dist, block_mask=block_mask)

        self.assertEqual(out_dt.full_tensor(), out)


class CPMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)


@flex_attention_hop.py_impl(CPMode)
@flex_attention_hop.py_impl(DTensor)
def cp_flex_attention(
    mode,
    query: DTensor,
    key: DTensor,
    value: DTensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[DTensor, DTensor]:
    print("Congrats! Flex attention is successfully dispatched!")

    assert isinstance(query, DTensor)
    assert isinstance(key, DTensor)
    assert isinstance(value, DTensor)
    q_local = query.to_local()
    k_local = key.full_tensor()
    v_local = value.full_tensor()
    device_type = q_local.device.type

    # extract context parallel mesh info
    cp_mesh = query.device_mesh
    assert cp_mesh.ndim == 1
    cp_rank = cp_mesh.get_local_rank()
    cp_group_size = cp_mesh.size()

    assert len(block_mask) == 13
    Q_LEN = block_mask[0]
    KV_LEN = block_mask[1]
    # kv_num_blocks = block_mask[2]
    # q_num_blocks = block_mask[6]
    mask_mod: _mask_mod_signature = block_mask[-1]
    Q_BLOCK_SIZE: int = block_mask[-3]
    KV_BLOCK_SIZE: int = block_mask[-2]
    # TODO: assume Q_BLOCK_SIZE == KV_BLOCK_SIZE
    assert Q_BLOCK_SIZE == KV_BLOCK_SIZE

    # TODO: assume no load-balancing for now, will add it later
    sharding_plan = regular_sharding(
        Q_LEN, KV_LEN, Q_BLOCK_SIZE, KV_BLOCK_SIZE, cp_group_size, device_type
    )

    # rewrite block_mask
    cp_mask_mod = rewrite_mask_mod_for_cp(
        mask_mod, cp_rank, Q_BLOCK_SIZE, sharding_plan
    )
    cp_block_mask = create_block_mask_cached(
        cp_mask_mod,
        B=1,
        H=1,
        M=Q_LEN // cp_group_size,
        N=KV_LEN,
        device=device_type,
        BLOCK_SIZE=(Q_BLOCK_SIZE, KV_BLOCK_SIZE),
    )

    out = flex_attention_hop(
        q_local,
        k_local,
        v_local,
        score_mod=score_mod,  # TODO: rewrite score_mod for cp
        block_mask=cp_block_mask.as_tuple(),
        scale=scale,
        kernel_options=kernel_options,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
    )

    return (
        DTensor.from_local(out[0], cp_mesh, [Shard(2)]),
        DTensor.from_local(out[1], cp_mesh, [Shard(2)]),
    )


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device, BLOCK_SIZE):
    block_mask = create_block_mask(
        score_mod, B, H, M, N, device=device, BLOCK_SIZE=BLOCK_SIZE
    )
    return block_mask


def regular_sharding(
    Q_LEN: int,
    KV_LEN: int,
    Q_BLOCK_SIZE: int,
    KV_BLOCK_SIZE: int,
    cp_size: int,
    device_type: str,
) -> torch.Tensor:
    assert Q_LEN == KV_LEN
    assert Q_BLOCK_SIZE == KV_BLOCK_SIZE
    assert Q_LEN % (Q_BLOCK_SIZE * cp_size) == 0
    q_num_blocks = Q_LEN // Q_BLOCK_SIZE
    local_num_blk = q_num_blocks // cp_size
    return torch.arange(q_num_blocks, device=device_type).view(cp_size, local_num_blk)


def rewrite_mask_mod_for_cp(
    mask_mod: _mask_mod_signature,
    rank: int,
    block_size: int,
    load_balancer_output: torch.Tensor,
) -> _mask_mod_signature:
    def local_q_idx_to_q_idx(local_q_idx) -> int:
        # calculate local block_idx and block_offset
        local_blk_idx, local_blk_offset = (
            local_q_idx // block_size,
            local_q_idx % block_size,
        )
        current_rank_blk_list = load_balancer_output[rank]
        blk_idx = current_rank_blk_list[local_blk_idx]
        return blk_idx * block_size + local_blk_offset

    return lambda b, h, q_idx, kv_idx: mask_mod(
        b, h, local_q_idx_to_q_idx(q_idx), kv_idx
    )


def shuffle_tensor_for_load_balancing(
    x: torch.Tensor, shuffle_tensor: torch.Tensor, dim: int
) -> torch.Tensor:
    # shuffle the tensor
    num_chunks = shuffle_tensor.numel()
    x_chunk_list = torch.chunk(x, num_chunks, dim=dim)
    assert len(x_chunk_list) == num_chunks
    new_x_chunk_list = [None] * num_chunks
    for blk_idx in range(num_chunks):
        new_blk_idx = shuffle_tensor[blk_idx].item()
        assert isinstance(new_blk_idx, int)
        new_x_chunk_list[blk_idx] = x_chunk_list[new_blk_idx]

    return torch.cat(new_x_chunk_list, dim=dim)


def interchange_index_value_2d(tensor: torch.Tensor) -> torch.Tensor:
    """
    Interchange the index and value in a PyTorch tensor. The input tensor has
    structure: rank -> [block_idx, ...] and the output tensor will be:
    block_idx -> block_idx_in_shuffled_tensor
    """
    flattened_tensor = tensor.view(-1)
    indices = torch.arange(flattened_tensor.numel(), device=flattened_tensor.device)
    revert_tensor = torch.empty_like(flattened_tensor)
    revert_tensor[flattened_tensor] = indices

    return revert_tensor


if __name__ == "__main__":
    run_tests()
