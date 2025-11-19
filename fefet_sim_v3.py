import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ===== Parameters (tunable) =====
levels = np.array([0.1, 0.4, 0.7, 1.0])  # desired normalized levels L0..L3
sigma_device = 0.02

# Device stress/healing model (gentle)
k_stress, alpha, beta = 5e-6, 1.3, 0.8   # trap growth scale and exponents
tau_relax = 15.0                         # rest-time relaxation
verify_tol = 0.02                        # verify stop tolerance
Ceff = 1.0                               # energy scale factor (arb.)
margin = 0.20                            # 20% spacing minimum (spec)

def f_program(vpulse, width):
    """Smooth saturation from a programming pulse -> 'nudge' in level."""
    s = 1.0 / (1.0 + np.exp(-0.8*(vpulse*width - 1.0)))
    return 0.15 * s

def update_trap(dv_trap, vpulse, width, t_rest):
    """Trap accumulation with stress; exponential relaxation with rest."""
    dv_trap = dv_trap + k_stress * (vpulse**alpha) * (width**beta)
    dv_trap = dv_trap * np.exp(-t_rest / tau_relax)
    return dv_trap

def write_with_verify(G_init, target, strategy="S3", max_iter=50, vpulse=2.0, width=1.0, t_rest=1.0):
    """
    Program toward 'target' from current level G_init.
    Returns: (G_final, dv_trap, verify_count, energy)
    """
    G = G_init
    dv_trap = 0.0
    verifies = 0
    energy = 0.0

    for _ in range(max_iter):
        if strategy == "S1":  # one-shot
            dG = f_program(vpulse, width)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += (vpulse**2) * width * Ceff
            dv_trap = update_trap(dv_trap, vpulse, width, 0.0)
            verifies += 1
            break

        elif strategy == "S2":  # multi-pulse, no verify
            dG = f_program(vpulse*0.7, width*0.5)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += ((vpulse*0.7)**2) * (width*0.5) * Ceff
            dv_trap = update_trap(dv_trap, vpulse*0.7, width*0.5, 0.0)
            verifies += 1
            # no verify criterion; rely on next cycle

        else:                   # S3: verify + adjust + rest (gentle)
            err = target - G
            if abs(err) <= verify_tol:
                break
            v_eff = 0.9 if err > 0 else 0.85
            w_eff = 0.4
            dG = f_program(v_eff, w_eff) * np.sign(err)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += (v_eff**2) * w_eff * Ceff
            dv_trap = update_trap(dv_trap, v_eff, w_eff, t_rest)
            verifies += 1

    return G, dv_trap, verifies, energy

def level_spacing_ok(last_seen_levels):
    """Check that adjacent stored levels remain separated by >= margin*nominal."""
    diffs = np.diff(np.sort(last_seen_levels))
    nominal = np.mean(diffs)
    # If we haven't yet written all four at least once, skip fail
    if np.any(np.isnan(last_seen_levels)):
        return True
    return np.all(diffs >= margin * nominal)

def simulate_cycles(n_cycles=1000, strategy="S3", check_every=50, verbose=False):
    """
    Cycle through L0->L1->L2->L3 repeating.
    Non-destructive check: track the last actually-written value for each level.
    """
    G = 0.0
    results = []
    last_seen = np.full(len(levels), np.nan)  # record of last stored G for each target

    if verbose:
        print(f"[RUN] strategy={strategy} | k_stress={k_stress} tau_relax={tau_relax} "
              f"verify_tol={verify_tol} margin={margin}")

    for c in range(n_cycles):
        idx = c % len(levels)
        tgt = levels[idx] + rng.normal(0, sigma_device)  # variability on target
        G, dv_trap, ver, E = write_with_verify(G, tgt, strategy=strategy)
        results.append((c, G, dv_trap, ver, E))
        last_seen[idx] = G  # update the last stored value for this target

        if (c+1) % check_every == 0:
            if not level_spacing_ok(last_seen):
                if verbose:
                    print(f"[FAIL] cycle={c+1} — level spacing collapsed (strategy={strategy})")
                return np.array(results), False

    if verbose:
        print(f"[PASS] completed {n_cycles} cycles (strategy={strategy})")
    return np.array(results), True

def run_and_report(n_cycles=1000, verbose=True):
    out = {}
    for strat in ["S1", "S2", "S3"]:
        res, ok = simulate_cycles(n_cycles, strategy=strat, verbose=verbose)
        mean_verify = float(np.mean(res[:,3])) if len(res) else 0.0
        total_energy = float(np.sum(res[:,4])) if len(res) else 0.0
        out[strat] = dict(ok=ok, cycles=len(res), mean_verify=mean_verify, total_energy=total_energy, res=res)
        print(f"{strat}: ok={ok} | cycles={len(res)} | mean_verify={mean_verify:.2f} | total_energy={total_energy:.2f}")
    return out

def plot_run(res, title_prefix="S3"):
    cycles = res[:,0]
    Gvals  = res[:,1]
    verifies = res[:,3]
    energy = res[:,4]
    cumE = np.cumsum(energy)

    # Level vs. cycle
    plt.figure()
    plt.plot(cycles, Gvals)
    plt.xlabel("Cycle")
    plt.ylabel("Stored Level (G)")
    plt.title(f"{title_prefix}: Level vs Cycle")
    plt.grid(True)

    # Verify count vs cycle
    plt.figure()
    plt.plot(cycles, verifies)
    plt.xlabel("Cycle")
    plt.ylabel("Verify count per write")
    plt.title(f"{title_prefix}: Verify Count vs Cycle")
    plt.grid(True)

    # Cumulative energy
    plt.figure()
    plt.plot(cycles, cumE)
    plt.xlabel("Cycle")
    plt.ylabel("Cumulative Energy (arb.)")
    plt.title(f"{title_prefix}: Cumulative Energy")
    plt.grid(True)

if __name__ == "__main__":
    summary = run_and_report(1000, verbose=True)

    # Plot S1, S2, S3 traces so you can compare visually
    for strat in ["S1", "S2", "S3"]:
        plot_run(summary[strat]["res"], title_prefix=strat)

    plt.show()
