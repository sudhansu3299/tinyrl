"""
Squares in Squares: simulated-annealing packer + strict validator + submission builder.

Method (per puzzle n):
  1. Start from the grid baseline (side = ceil(sqrt(n))), always valid.
  2. Shrink the container a little, scaling centers; this creates overlaps.
  3. Run simulated annealing on an energy = sum(penetration^2) + sum(boundary violation^2).
     Energy 0 means a valid packing. If found, keep it and shrink again; otherwise
     restore the last valid packing and use a smaller shrink step.
  4. Squares are optimised slightly inflated (half-size 0.5 + margin), so the final
     unit-square packing has a real gap and survives the scorer's floating-point checks.

ASSUMPTIONS to confirm on the competition's Evaluation tab:
  - score uses the side of the axis-aligned bounding square of all squares in a puzzle
  - the container is anchored at the origin (we shift each puzzle so min corner = (0, 0))
  - angle is a rotation in radians about the square's center

Usage:
  python squares.py solve --ns 5 10 11 --seconds 30
  python squares.py submit --max-n 40 --seconds 20 --out submission.csv
  python squares.py submit --max-n 40 --seconds 60 --seed 1 --merge --out submission.csv
"""
import argparse
import csv
import math
import os
import time
from multiprocessing import Pool

import numpy as np
from numba import njit

MARGIN = 1e-6          # inflation used while optimising (gives a real gap in the final answer)
VALID_TOL = 1e-9       # validator: overlaps deeper than this are rejected


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def grid_packing(n):
    """Baseline row-by-row grid, identical to the sample submission."""
    k = math.ceil(math.sqrt(n))
    idx = np.arange(n)
    x = (idx % k) + 0.5
    y = (idx // k) + 0.5
    return x, y, np.zeros(n)


def bbox(x, y, t, hs=0.5):
    """Axis-aligned bounding box of rotated squares with half-size hs."""
    h = hs * (np.abs(np.cos(t)) + np.abs(np.sin(t)))
    return (x - h).min(), (x + h).max(), (y - h).min(), (y + h).max()


def normalize(x, y, t, hs=0.5):
    """Shift so the bounding box starts at (0, 0). Returns shifted arrays and side length."""
    x0, x1, y0, y1 = bbox(x, y, t, hs)
    return x - x0, y - y0, t, max(x1 - x0, y1 - y0)


def _min_overlap(dx, dy, ti, tj, hs):
    """SAT: minimum overlap over the 4 candidate separating axes (<=0 means separated)."""
    ci, si = np.cos(ti), np.sin(ti)
    cj, sj = np.cos(tj), np.sin(tj)
    phi = tj - ti
    R = hs * (1.0 + np.abs(np.cos(phi)) + np.abs(np.sin(phi)))  # same sum of radii on every axis
    p1 = np.abs(dx * ci + dy * si)
    p2 = np.abs(-dx * si + dy * ci)
    p3 = np.abs(dx * cj + dy * sj)
    p4 = np.abs(-dx * sj + dy * cj)
    return R - np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))


def is_valid(x, y, t, tol=VALID_TOL):
    """Strict pairwise check on true unit squares."""
    n = len(x)
    for i in range(n - 1):
        ov = _min_overlap(x[i + 1:] - x[i], y[i + 1:] - y[i], t[i], t[i + 1:], 0.5)
        if ov.max() > tol:
            return False
    return True


# ----------------------------------------------------------------------------
# Energy + simulated annealing (compiled with numba)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _sq_energy(i, xi, yi, ti, x, y, t, L, hs):
    """Energy of square i at (xi, yi, ti) against all others plus the walls."""
    n = x.shape[0]
    ci, si = math.cos(ti), math.sin(ti)
    e = 0.0
    for j in range(n):
        if j == i:
            continue
        dx, dy = x[j] - xi, y[j] - yi
        if dx * dx + dy * dy > 8.01 * hs * hs:             # centers too far to touch
            continue
        cj, sj = math.cos(t[j]), math.sin(t[j])
        phi = t[j] - ti
        R = hs * (1.0 + abs(math.cos(phi)) + abs(math.sin(phi)))
        p = max(max(abs(dx * ci + dy * si), abs(-dx * si + dy * ci)),
                max(abs(dx * cj + dy * sj), abs(-dx * sj + dy * cj)))
        ov = R - p
        if ov > 0.0:
            e += ov * ov
    h = hs * (abs(ci) + abs(si))
    e += max(0.0, h - xi) ** 2 + max(0.0, xi + h - L) ** 2
    e += max(0.0, h - yi) ** 2 + max(0.0, yi + h - L) ** 2
    return e


@njit(cache=True)
def _total_energy(x, y, t, L, hs):
    """Full energy (pairs counted once, walls once)."""
    n = x.shape[0]
    e = 0.0
    for i in range(n):
        h = hs * (abs(math.cos(t[i])) + abs(math.sin(t[i])))
        e += max(0.0, h - x[i]) ** 2 + max(0.0, x[i] + h - L) ** 2
        e += max(0.0, h - y[i]) ** 2 + max(0.0, y[i] + h - L) ** 2
        for j in range(i + 1, n):
            dx, dy = x[j] - x[i], y[j] - y[i]
            phi = t[j] - t[i]
            R = hs * (1.0 + abs(math.cos(phi)) + abs(math.sin(phi)))
            ci, si = math.cos(t[i]), math.sin(t[i])
            cj, sj = math.cos(t[j]), math.sin(t[j])
            p = max(max(abs(dx * ci + dy * si), abs(-dx * si + dy * ci)),
                    max(abs(dx * cj + dy * sj), abs(-dx * sj + dy * cj)))
            ov = R - p
            if ov > 0.0:
                e += ov * ov
    return e


@njit(cache=True)
def _anneal_nb(x, y, t, L, hs, iters, T0, T1, step0, seed, q, epsilon0):
    """Q-learning-guided simulated annealing.

    State = energy regime x temperature phase x recent trend x acceptance regime (72 states).
    Actions = translation, rotation-only, combined move.
    SA still controls Metropolis acceptance; tabular Q-learning learns which
    move type tends to reduce energy in each coarse state.
    """
    np.random.seed(seed)
    n = x.shape[0]
    E = _total_energy(x, y, t, L, hs)
    if E == 0.0:
        return True
    cool = (T1 / T0) ** (1.0 / iters)
    T = T0

    # Shared Q-table is passed by the caller and persists across episodes.
    alpha = 0.10
    gamma = 0.90
    epsilon = epsilon0
    eps_decay = 0.9995
    best_E = E
    trend_ema = 0.0
    accepted_window = 0
    window_count = 0

    for it in range(iters):
        # State features: energy density, temperature, recent energy trend,
        # and recent acceptance rate. The richer state can distinguish a
        # stagnant search from one that is steadily making progress.
        e_density = E / n
        if e_density < 0.01:
            e_bin = 0
        elif e_density < 0.10:
            e_bin = 1
        elif e_density < 0.50:
            e_bin = 2
        else:
            e_bin = 3
        temp_ratio = T / T0
        if temp_ratio > 0.30:
            t_bin = 0
        elif temp_ratio > 0.03:
            t_bin = 1
        else:
            t_bin = 2
        if trend_ema > 1e-4:
            trend_bin = 0  # energy tends to be increasing
        elif trend_ema < -1e-4:
            trend_bin = 2  # energy tends to be decreasing
        else:
            trend_bin = 1  # mostly flat / stagnant
        accept_rate = accepted_window / max(window_count, 1)
        accept_bin = 0 if accept_rate < 0.20 else 1
        state = ((e_bin * 3 + t_bin) * 3 + trend_bin) * 2 + accept_bin

        # Epsilon-greedy action selection, with random tie-breaking.
        if np.random.random() < epsilon:
            op = np.random.randint(3)
        else:
            best_q = q[state, 0]
            for a in range(1, 3):
                if q[state, a] > best_q:
                    best_q = q[state, a]
            ties = np.zeros(3, dtype=np.int64)
            nties = 0
            for a in range(3):
                if q[state, a] >= best_q - 1e-15:
                    ties[nties] = a
                    nties += 1
            op = ties[np.random.randint(nties)]

        i = np.random.randint(n)
        sc = max(step0 * math.sqrt(T / T0), 2e-4)
        xn, yn, tn = x[i], y[i], t[i]
        if op == 0:  # translation
            xn = x[i] + np.random.normal(0.0, sc)
            yn = y[i] + np.random.normal(0.0, sc)
        elif op == 1:  # rotation
            tn = t[i] + np.random.normal(0.0, 2.0 * sc)
        else:  # combined translation + rotation
            xn = x[i] + np.random.normal(0.0, sc)
            yn = y[i] + np.random.normal(0.0, sc)
            tn = t[i] + np.random.normal(0.0, 2.0 * sc)

        old_E = E
        old_best_E = best_E
        old_local = _sq_energy(i, x[i], y[i], t[i], x, y, t, L, hs)
        new_local = _sq_energy(i, xn, yn, tn, x, y, t, L, hs)
        d = new_local - old_local
        accepted = d <= 0.0 or np.random.random() < math.exp(-d / T)
        if accepted:
            x[i], y[i], t[i] = xn, yn, tn
            E += d
            if E < 1e-12:
                E = _total_energy(x, y, t, L, hs)
                if E == 0.0:
                    return True

        # Reward actual global-energy progress; penalize accepted uphill moves
        # gently, while SA remains free to accept them to escape local minima.
        if accepted:
            new_E = E
            accepted_window += 1
        else:
            new_E = old_E
        window_count += 1
        if window_count >= 100:
            accepted_window = int(accepted_window * 0.5)
            window_count = max(1, int(window_count * 0.5))

        # Reward immediate global progress plus a smaller bonus for reaching
        # a new best energy. SA can still accept uphill moves; the reward only
        # trains future move selection and does not override Metropolis.
        denom = max(old_E, 1e-9)
        immediate = (old_E - new_E) / denom
        if E < best_E:
            best_E = E
        best_gain = (old_best_E - best_E) / max(old_best_E, 1e-9)
        reward = immediate + 0.25 * best_gain
        if reward > 1.0:
            reward = 1.0
        elif reward < -1.0:
            reward = -1.0
        trend_ema = 0.95 * trend_ema + 0.05 * (new_E - old_E) / denom

        # Next-state estimate for Q-learning.
        next_density = E / n
        if next_density < 0.01:
            ne_bin = 0
        elif next_density < 0.10:
            ne_bin = 1
        elif next_density < 0.50:
            ne_bin = 2
        else:
            ne_bin = 3
        next_ratio = (T * cool) / T0
        if next_ratio > 0.30:
            nt_bin = 0
        elif next_ratio > 0.03:
            nt_bin = 1
        else:
            nt_bin = 2
        if trend_ema > 1e-4:
            next_trend_bin = 0
        elif trend_ema < -1e-4:
            next_trend_bin = 2
        else:
            next_trend_bin = 1
        next_accept_rate = accepted_window / max(window_count, 1)
        next_accept_bin = 0 if next_accept_rate < 0.20 else 1
        next_state = ((ne_bin * 3 + nt_bin) * 3 + next_trend_bin) * 2 + next_accept_bin
        max_next_q = q[next_state, 0]
        for a in range(1, 3):
            if q[next_state, a] > max_next_q:
                max_next_q = q[next_state, a]
        q[state, op] += alpha * (reward + gamma * max_next_q - q[state, op])
        epsilon = max(0.03, epsilon * eps_decay)

        T *= cool
        if it % 20000 == 19999:
            E = _total_energy(x, y, t, L, hs)
    return False


def anneal(x, y, t, L, hs, rng, iters, T0=0.1, T1=1e-7, step0=0.5, deadline=None, policy=None, epsilon=0.10):
    """Python wrapper: returns (x, y, t, success). Works on copies."""
    x, y, t = x.copy(), y.copy(), t.copy()
    if policy is None:
        policy = np.zeros((72, 3), dtype=np.float64)
    ok = _anneal_nb(x, y, t, float(L), float(hs), int(iters), float(T0), float(T1),
                    float(step0), int(rng.integers(1 << 31)), policy, float(epsilon))
    return x, y, t, ok


def optimize_n(n, seconds=30.0, seed=0, iters=None, T_hot=0.1, policy=None, epsilon=0.10):
    """Find a small valid packing for n unit squares. Returns (x, y, t, side).

    Alternates two kinds of attempts at a target container size L = best - delta:
      * hot random restart (escapes the grid's basin; this is how tilted packings appear)
      * warm start from the best packing so far (squeezes it a bit further)
    On repeated failure delta is halved; on success the best packing is updated.
    """
    hs = 0.5 + MARGIN
    rng = np.random.default_rng(seed)
    k = math.ceil(math.sqrt(n))
    gx, gy, gt = grid_packing(n)
    if k * k == n:                       # perfect squares: grid is optimal
        return gx, gy, gt, float(k)
    iters = iters or 30000 * n
    sp = (k + 0.02) / k                  # spread the grid slightly so inflated squares are feasible
    bx, by, bt, bL = normalize(gx * sp, gy * sp, gt, hs)

    deadline = time.time() + seconds
    delta = 0.25 * (k - math.sqrt(n))
    fails, attempt = 0, 0
    while delta > 3e-3 and time.time() < deadline:
        L = bL - delta
        if attempt % 2 == 0:             # hot random restart
            x = rng.uniform(0.6, L - 0.6, n)
            y = rng.uniform(0.6, L - 0.6, n)
            t = rng.uniform(0, math.pi / 2, n)
            res = anneal(x, y, t, L, hs, rng, iters, T0=T_hot, T1=1e-7, step0=0.5, deadline=deadline, policy=policy, epsilon=epsilon)
        else:                            # warm start from best
            sc = L / bL
            res = anneal(bx * sc, by * sc, bt + rng.normal(0, 0.02, n), L, hs, rng, iters,
                         T0=0.01, T1=1e-8, step0=0.1, deadline=deadline, policy=policy, epsilon=epsilon)
        attempt += 1
        if res[3]:
            bx, by, bt, bL = normalize(res[0], res[1], res[2], hs)
            fails = 0
        else:
            fails += 1
            if fails >= 4:
                delta *= 0.5
                fails = 0

    x, y, t, true_side = normalize(bx, by, bt, 0.5)   # report side for real unit squares
    return x, y, t, true_side


# ----------------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------------
def score_n(n, side):
    """Competition metric for one puzzle: L / sqrt(n) - 1 (lower is better)."""
    return side / math.sqrt(n) - 1.0


def _worker(args):
    n, seconds, seed = args
    x, y, t, side = optimize_n(n, seconds, seed)
    return n, x, y, t, side


def solve_many(ns, seconds, restarts=1, workers=None, seed=0):
    """Optimise each n (in parallel). Same seed + more seconds/restarts never does worse."""
    jobs = [(n, seconds, seed * 1_000_000 + 1000 * r + n) for n in ns for r in range(restarts)]
    best = {}
    with Pool(workers) as pool:
        for n, x, y, t, side in pool.imap_unordered(_worker, jobs):
            if n not in best or side < best[n][3]:
                best[n] = (x, y, t, side)
    return best


def load_submission(path):
    """Read an existing submission and return {n: (x, y, t, side)} for puzzles beating the grid."""
    out, rows = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r["n"]), []).append((float(r["x"]), float(r["y"]), float(r["angle"])))
    for n, a in rows.items():
        x, y, t = np.array(a).T
        x0, x1, y0, y1 = bbox(x, y, t)
        side = max(x1 - x0, y1 - y0)
        if side < math.ceil(math.sqrt(n)) - 1e-9 and is_valid(x, y, t):
            out[n] = (x - x0, y - y0, t, side)
    return out


def write_submission(results, path, max_n=1000):
    """Write the CSV. Any n without a validated improvement falls back to the grid.
    Returns (number improved, total leaderboard score, grid-baseline score)."""
    row_id, improved, total, baseline = 0, 0, 0.0, 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for n in range(1, max_n + 1):
            x, y, t = grid_packing(n)
            base = math.ceil(math.sqrt(n))
            side = float(base)
            if n in results:
                rx, ry, rt, rside = results[n]
                if rside < base - 1e-9 and is_valid(rx, ry, rt):
                    x, y, t, side = rx, ry, rt, rside
                    improved += 1
            total += score_n(n, side)
            baseline += score_n(n, base)
            for s in range(n):
                w.writerow([row_id, n, s, repr(float(x[s])), repr(float(y[s])), repr(float(t[s]))])
                row_id += 1
    return improved, total, baseline


def train_policy(episodes=1000, out="rl_policy.npz", seed=0, max_n=30, seconds_per_episode=0.25):
    """Train one shared Q-table across many episodes and puzzle sizes, then save it."""
    rng = np.random.default_rng(seed)
    q = np.zeros((72, 3), dtype=np.float64)
    successes = 0
    for ep in range(episodes):
        # Mix puzzle sizes and seeds so the policy does not see only one layout.
        n = int(rng.integers(2, max_n + 1))
        if math.isqrt(n) ** 2 == n:
            n = 2 if n == max_n else n + 1
        _, _, _, side = optimize_n(
            n, seconds=seconds_per_episode,
            seed=int(rng.integers(1 << 31)),
            iters=max(500, 300 * n), policy=q, epsilon=max(0.05, 0.8 * (1 - ep / max(episodes, 1)))
        )
        # optimize_n returns a valid best layout; use successful learned episodes as a basic diagnostic.
        if side < math.ceil(math.sqrt(n)) - 1e-9:
            successes += 1
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"episode {ep+1}/{episodes}; improved puzzles in episodes: {successes}")
    np.savez(out, q=q, version=np.array([2]), seed=np.array([seed]),
             state_description=np.array(["energy4_temp3_trend3_acceptance2"]))
    print(f"saved shared Q-policy to {out}; successful episodes: {successes}/{episodes}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="train and save a shared Q-learning policy")
    tr.add_argument("--episodes", type=int, default=1000)
    tr.add_argument("--max-n", type=int, default=30)
    tr.add_argument("--seconds-per-episode", type=float, default=0.25)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", default="rl_policy.npz")
    s = sub.add_parser("solve", help="optimise a few puzzles and print their scores")
    s.add_argument("--ns", type=int, nargs="+", required=True)
    s.add_argument("--seconds", type=float, default=30)
    s.add_argument("--restarts", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    u = sub.add_parser("submit", help="optimise n = 2..max-n and write the full CSV")
    u.add_argument("--policy", default=None, help="load a trained policy .npz")
    u.add_argument("--max-n", type=int, default=40, help="optimise n = 2..max-n; rest stay as grid")
    u.add_argument("--seconds", type=float, default=20)
    u.add_argument("--restarts", type=int, default=1)
    u.add_argument("--seed", type=int, default=0, help="change this to explore different random starts")
    u.add_argument("--out", default="submission.csv")
    u.add_argument("--merge", action="store_true",
                   help="keep the best packing per puzzle between --out (if it exists) and this run")
    a = ap.parse_args()

    if a.cmd == "train":
        train_policy(a.episodes, a.out, a.seed, a.max_n, a.seconds_per_episode)
        return

    if a.cmd == "solve":
        res = solve_many(a.ns, a.seconds, a.restarts, seed=a.seed)
        total = 0.0
        for n in sorted(res):
            x, y, t, side = res[n]
            g = math.ceil(math.sqrt(n))
            use = side if (side < g - 1e-9 and is_valid(x, y, t)) else g
            total += score_n(n, use)
            print(f"n={n:4d}  side={use:.6f}  score={score_n(n, use):.5f}  "
                  f"(grid side={g}, grid score={score_n(n, g):.5f})  valid={is_valid(x, y, t)}")
        print(f"sum of scores for these puzzles: {total:.5f}")
    else:
        ns = [n for n in range(2, a.max_n + 1) if math.isqrt(n) ** 2 != n]
        if a.policy:
            data = np.load(a.policy)
            policy = data["q"].astype(np.float64)
            if policy.shape != (72, 3):
                raise ValueError(
                    f"Policy shape {policy.shape} is incompatible with this version; "
                    "retrain using this script to create a 72x3 policy."
                )
            res = {}
            for n in ns:
                best = None
                for r in range(a.restarts):
                    candidate = optimize_n(n, a.seconds, a.seed * 1_000_000 + 1000 * r + n, policy=policy, epsilon=0.03)
                    if best is None or candidate[3] < best[3]:
                        best = candidate
                res[n] = best
        else:
            res = solve_many(ns, a.seconds, a.restarts, seed=a.seed)
        if a.merge and os.path.exists(a.out):
            old = load_submission(a.out)
            for n, v in old.items():
                if n not in res or v[3] < res[n][3]:
                    res[n] = v
            print(f"merged with existing {a.out} ({len(old)} improved puzzles there)")
        k, total, baseline = write_submission(res, a.out)
        print(f"wrote {a.out}; improved {k} of {len(ns)} non-square puzzles over the grid")
        print(f"leaderboard score (lower is better): {total:.4f}   grid baseline: {baseline:.4f}   "
              f"gain: {baseline - total:.4f}")


if __name__ == "__main__":
    main()
