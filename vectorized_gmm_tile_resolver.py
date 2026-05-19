import platform
import sys

import torch

IS_NOT_WINDOWS = platform.system() != "Windows"


if IS_NOT_WINDOWS:
    import triton
    import triton.language as tl


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


DEVICE = "cuda" if IS_NOT_WINDOWS else "cpu"
DTYPE = torch.int64
M = 267424
N = 2560
# original G is 32, testing with an odd / non-power-of-2 G
G = 13  # 32
G_POW2 = next_power_of_2(G)

BLOCK_SIZE_M = 256
BLOCK_SIZE_N = 256


def gen_group_sizes():
    gs_list = [M // G] * G
    gs_list[-1] += M % G
    group_sizes = torch.tensor(gs_list, dtype=DTYPE, device=DEVICE)
    assert torch.all(group_sizes >= 0), "all group_sizes must be >= 0"
    assert torch.sum(group_sizes) == M, "group_sizes must add up to M"
    return group_sizes


def num_tiles(group_sizes):
    num_m_tiles = cdiv(group_sizes, BLOCK_SIZE_M)
    num_n_tiles = cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles
    return torch.sum(num_tiles).item()


DEBUG = False


def debug(x_name, x):
    if DEBUG:
        print(x_name, x)


def torch_resolve_tile(tile, group_sizes):
    debug("tile", tile)
    assert tile >= 0, "tile must be >= 0"

    g_range = torch.arange(G_POW2, device=DEVICE)
    g_mask = g_range < G
    zeros = torch.zeros((G_POW2,), dtype=DTYPE, device=DEVICE)

    # emulate tl.load, with a power-of-2 tensor
    loaded_group_sizes = zeros.clone()
    loaded_group_sizes[:G] = group_sizes
    assert torch.all(loaded_group_sizes >= 0), "all group_sizes must be >= 0"
    assert torch.sum(loaded_group_sizes) == M, "group_sizes must add up to M"
    debug("group_sizes", loaded_group_sizes)

    num_m_tiles = cdiv(loaded_group_sizes, BLOCK_SIZE_M)
    debug("num_m_tiles", num_m_tiles)
    num_n_tiles = cdiv(N, BLOCK_SIZE_N)
    debug("num_n_tiles", num_n_tiles)
    num_tiles = num_m_tiles * num_n_tiles
    debug("num_tiles", num_tiles)
    cumsum_tile = torch.where(g_mask, torch.cumsum(num_tiles, dim=0), zeros)
    debug("cumsum_tile", cumsum_tile)

    max_tile = torch.max(cumsum_tile)
    debug("max_tile", max_tile.item())
    assert tile < max_tile, f"tile must be < {max_tile}"

    cumsum_m = torch.where(g_mask, torch.cumsum(loaded_group_sizes, dim=0), zeros)
    debug("cumsum_m", cumsum_m)
    g = torch.sum((cumsum_tile <= tile) & g_mask)
    debug("g", g.item())
    prev_mask = g_range < g
    debug("prev_mask", prev_mask)
    prev_cumsum_m = torch.max(torch.where(prev_mask, cumsum_m, zeros))
    debug("prev_cumsum_m", prev_cumsum_m.item())
    prev_cumsum_tile = torch.max(torch.where(prev_mask, cumsum_tile, zeros))
    debug("prev_cumsum_tile", prev_cumsum_tile.item())
    g_cumsum_m = torch.max(torch.where(g_range == g, cumsum_m, zeros))
    debug("g_cumsum_m", g_cumsum_m.item())

    m = g_cumsum_m - prev_cumsum_m
    num_m_tiles_out = cdiv(m, BLOCK_SIZE_M)
    tile_in_mm = tile - prev_cumsum_tile

    debug("out: g>", g.item())
    debug("out: m>", m.item())
    debug("out: num_m_tiles>", num_m_tiles_out.item())
    debug("out: last_m>", prev_cumsum_m.item())
    debug("out: tile_in_mm>", tile_in_mm.item())
    return (
        g.item(),
        m.item(),
        num_m_tiles_out.item(),
        prev_cumsum_m.item(),
        tile_in_mm.item(),
    )


def triton_resolve_tile(tile, group_sizes):
    assert IS_NOT_WINDOWS, "Triton tile resolver should not be called on Windows"
    assert group_sizes.is_cuda, "group_sizes tensor must be on GPU"

    @triton.jit
    def triton_resolve_tile_fn(
        group_sizes_ptr,
        tile,
        G,
        N,
        BLOCK_SIZE_G: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        tl.device_assert(tile >= 0, "tile < 0")
        int_type = group_sizes_ptr.type.element_ty
        g_range = tl.arange(0, BLOCK_SIZE_G)
        g_mask = g_range < G
        zeros = tl.zeros((BLOCK_SIZE_G,), int_type)
        group_sizes = tl.load(group_sizes_ptr + g_range, mask=g_mask, other=0)
        num_m_tiles = tl.cdiv(group_sizes, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N).to(int_type)
        tl.device_assert(num_n_tiles > 0, "num_n_tiles <= 0")
        num_tiles = num_m_tiles * num_n_tiles
        cumsum_tile = tl.where(g_mask, tl.cumsum(num_tiles, dtype=int_type), zeros)
        max_tile = tl.max(cumsum_tile)
        tl.device_assert(tile < max_tile, "tile >= max_tile")
        cumsum_m = tl.where(g_mask, tl.cumsum(group_sizes, dtype=int_type), zeros)
        g = tl.sum((cumsum_tile <= tile) & g_mask, dtype=int_type)
        prev_mask = g_range < g
        prev_cumsum_m = tl.max(tl.where(prev_mask, cumsum_m, zeros))
        prev_cumsum_tile = tl.max(tl.where(prev_mask, cumsum_tile, zeros))
        g_cumsum_m = tl.max(tl.where(g_range == g, cumsum_m, zeros))
        m = g_cumsum_m - prev_cumsum_m
        num_m_tiles_out = tl.cdiv(m, BLOCK_SIZE_M)
        tile_in_mm = tile - prev_cumsum_tile
        #      g, m, num_m_tiles,     last_m,        tile_in_mm
        return g, m, num_m_tiles_out, prev_cumsum_m, tile_in_mm

    @triton.jit
    def triton_resolve_tile_kernel(
        # input tensors
        tile_ptr,
        group_sizes_ptr,
        # output tensors
        g_ptr,
        m_ptr,
        num_m_tiles_ptr,
        last_m_ptr,
        tile_in_mm_ptr,
        # shape
        G,
        N,
        # meta-parameters
        BLOCK_SIZE_G: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        tl.assume(N > 0)
        tl.assume(G > 0)
        tile = tl.load(tile_ptr)
        g, m, num_m_tiles, last_m, tile_in_mm = triton_resolve_tile_fn(
            group_sizes_ptr,
            tile,
            G,
            N,
            BLOCK_SIZE_G=BLOCK_SIZE_G,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        tl.store(g_ptr, g)
        tl.store(m_ptr, m)
        tl.store(num_m_tiles_ptr, num_m_tiles)
        tl.store(last_m_ptr, last_m)
        tl.store(tile_in_mm_ptr, tile_in_mm)

    d_tile = torch.tensor((tile,), dtype=DTYPE, device=DEVICE)
    d_g = torch.empty((1,), dtype=DTYPE, device=DEVICE)
    d_m = torch.empty((1,), dtype=DTYPE, device=DEVICE)
    d_num_m_tiles = torch.empty((1,), dtype=DTYPE, device=DEVICE)
    d_last_m = torch.empty((1,), dtype=DTYPE, device=DEVICE)
    d_tile_in_mm = torch.empty((1,), dtype=DTYPE, device=DEVICE)

    triton_resolve_tile_kernel[(1,)](
        # input tensors
        d_tile,
        group_sizes,
        # output tensors
        d_g,
        d_m,
        d_num_m_tiles,
        d_last_m,
        d_tile_in_mm,
        # shape
        G,
        N,
        # meta-parameters
        BLOCK_SIZE_G=G_POW2,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_M,
    )

    return (
        d_g.item(),
        d_m.item(),
        d_num_m_tiles.item(),
        d_last_m.item(),
        d_tile_in_mm.item(),
    )


def check_title_resolution(tile, group_sizes):
    torch_g, torch_m, torch_num_m_tiles, torch_last_m, torch_tile_in_mm = (
        torch_resolve_tile(tile, group_sizes)
    )
    if IS_NOT_WINDOWS:
        triton_g, triton_m, triton_num_m_tiles, triton_last_m, triton_tile_in_mm = (
            triton_resolve_tile(tile, group_sizes)
        )
        if DEBUG:
            print("Triton out:")
            print(
                triton_g, triton_m, triton_num_m_tiles, triton_last_m, triton_tile_in_mm
            )
        assert torch_g == triton_g, f"tile={tile}: g mismatch"
        assert torch_m == triton_m, f"tile={tile}: m mismatch"
        assert (
            torch_num_m_tiles == triton_num_m_tiles
        ), f"tile={tile}: num_m_tiles mismatch"
        assert torch_last_m == triton_last_m, f"tile={tile}: last_m mismatch"
        assert (
            torch_tile_in_mm == triton_tile_in_mm
        ), f"tile={tile}: tile_in_mm mismatch"


if __name__ == "__main__":
    group_sizes = gen_group_sizes()
    tiles_to_iter = (
        (int(sys.argv[1]),) if len(sys.argv) > 1 else range(0, num_tiles(group_sizes))
    )
    for tile in tiles_to_iter:
        check_title_resolution(tile, group_sizes)
