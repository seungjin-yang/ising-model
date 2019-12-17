import os
import argparse
import numpy as np
import numba


NEIGHBORHOOD_DISTANCE = [(1, 0), (-1, 0), (0, 1), (0, -1)]
CRITICAL_TEMPERATURE = 2 / np.log(1 + np.sqrt(2))


def initialize_lattice(size):
    # FIXME rename
    size = (size, size)
    return 2 * np.random.randint(low=0, high=2, size=size, dtype=np.int8) - 1


@numba.njit
def get_neighbors(i, j, L):
    # FIXME L
    # FIXME neighborhood distance
    return [((i + di) % L, (j + dj) % L) for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]]


@numba.njit
def update(lattice, L, p):
    i, j = np.random.randint(low=0, high=L, size=2)
    spin = lattice[i, j]

    candidates = get_neighbors(i, j, L)
    cluster = [(i, j)]

    while len(candidates) > 0:
        neighbor = candidates.pop(0)
        if neighbor in cluster:
            continue

        if spin == lattice[neighbor] and p > np.random.rand():
            cluster.append(neighbor)
            candidates += get_neighbors(neighbor[0], neighbor[1], L)

    for each in cluster:
        lattice[each] *= -1


@numba.njit
def compute_magnetization(lattice):
    return lattice.sum()


@numba.njit
def compute_susceptibility(M, N, T, H):
    if H == 0:
        M = np.abs(M)
    return M.var() / (N * T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lattice-size', type=int, default=64)
    parser.add_argument('-t', '--temperature-scale', type=float, default=1.0)
    parser.add_argument('-H', '--external-field', type=float, default=0)
    parser.add_argument('-j', '--coupling-constant', type=float, default=1)
    parser.add_argument('-s', '--mc-steps', type=int, default=1)
    parser.add_argument('-n', '--num-ensemble', type=int, default=1000)
    parser.add_argument('-o', '--out-dir', type=str, default='./data/')
    args = parser.parse_args()

    ############################################################################
    # NOTE
    ############################################################################
    temperature = args.temperature_scale * CRITICAL_TEMPERATURE
    beta = 1 / temperature
    p = 1 - np.exp(-2 * beta * args.coupling_constant)
    lattice = initialize_lattice(args.lattice_size)
    num_sites = np.prod(lattice.shape)

    ############################################################################
    # NOTE
    ############################################################################
    for step in range(1, args.mc_steps + 1):
        print(f'{step} / {args.mc_steps}')
        for _ in range(num_sites):
            update(lattice, args.lattice_size, p)

    ############################################################################
    # NOTE
    ############################################################################
    magnetizations = []
    for _ in range(args.num_ensemble):
        update(lattice, args.lattice_size, p)
        M = compute_magnetization(lattice)
        magnetizations.append(M)
    magnetizations = np.array(magnetizations)

    susceptibility = compute_susceptibility(
        M=magnetizations,
        N=num_sites,
        T=temperature,
        H=args.external_field)

    ############################################################################
    # NOTE
    ############################################################################
    basename = f'ising-wolff'
    basename += f'_L-{args.lattice_size}'
    basename += f'_ToverTc-{args.temperature_scale:.4f}'
    basename += f'_H-{args.external_field}'
    basename += '.npz'
    path = os.path.join(args.out_dir, basename)

    np.savez(
        path,
        magnetizations=magnetizations,
        susceptibility=susceptibility)


if __name__ == '__main__':
    main()
