import numpy as np

rng = np.random.default_rng(0)

# --- Parameters (gentler, with stronger healing) ---
levels = np.array([0.1, 0.4, 0.7, 1.0])
sigma_device = 0.02

# CHANGED ↓ (tune endurance)
k_stress, alpha, beta = 5e-6, 1.3, 0.8     # was 2e-4,1.3,0.8 in baseline
tau_relax = 15.0                            # was 5.0 in baseline
verify_tol = 0.02                           # was 0.03 in baseline
Ceff = 1.0
margin = 0.20                               # was 0.25 in baseline

def f_program(vpulse, width):
    s = 1.0 / (1.0 + np.exp(-0.8*(vpulse*width - 1.0)))
    return 0.15 * s

def update_trap(dv_trap, vpulse, width, t_rest):
    dv_trap = dv_trap + k_stress * (vpulse**alpha) * (width**beta)
    dv_trap = dv_trap * np.exp(-t_rest / tau_relax)
    return dv_trap

def write_with_verify(G_init, target, strategy="S3", max_iter=50, vpulse=2.0, width=1.0, t_rest=1.0):
    """Return final level, trap shift, verify count, energy."""
    G = G_init
    dv_trap = 0.0
    verifies = 0
    energy = 0.0
    for i in range(max_iter):
        if strategy == "S1":  # one-shot (baseline)
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

        else:  # S3: verify + adjust + rest
            err = target - G
            if abs(err) <= verify_tol:
                break

            # CHANGED ↓ gentler corrective pulses + longer rest
            v_eff = 0.9 if err > 0 else 0.85   # was 1.2/0.9
            w_eff = 0.4                       # was 0.6
            dG = f_program(v_eff, w_eff) * np.sign(err)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += (v_eff**2) * w_eff * Ceff

            # CHANGED ↓ rest time increased to promote healing
            dv_trap = update_trap(dv_trap, v_eff, w_eff, t_rest)  # t_rest default now 1.0

        verifies += 1
    return G, dv_trap, verifies, energy

def level_spacing_ok(G_levels):
    diffs = np.diff(np.sort(G_levels))
    nominal = np.mean(diffs)
    return np.all(diffs >= margin*nominal)

def simulate_cycles(n_cycles=1000, strategy="S3", check_every=50, verbose=False):
    G = 0.0
    results = []
    if verbose:
        print(f"[RUN] strategy={strategy} | k_stress={k_stress}  tau_relax={tau_relax}  "
              f"verify_tol={verify_tol}  margin={margin}")
    for c in range(n_cycles):
        tgt = levels[c % len(levels)] + rng.normal(0, sigma_device)
        G, dv_trap, ver, E = write_with_verify(G, tgt, strategy=strategy)
        results.append((c, G, dv_trap, ver, E))

        if (c+1) % check_every == 0:
            # probe stability across the four levels
            probe = []
            G_probe = G
            for t in levels:
                G_probe, _, _, _ = write_with_verify(G_probe, t, strategy=strategy, max_iter=10)
                probe.append(G_probe)
            if not level_spacing_ok(np.array(probe)):
                if verbose:
                    print(f"[FAIL] cycle={c+1} — level spacing collapsed (strategy={strategy})")
                return np.array(results), False
    if verbose:
        print(f"[PASS] completed {n_cycles} cycles (strategy={strategy})")
    return np.array(results), True

if __name__ == "__main__":
    # Quick comparison table for your report
    for strat in ["S1", "S2", "S3"]:
        res, ok = simulate_cycles(1000, strategy=strat, verbose=True)
        mean_verify = float(np.mean(res[:,3])) if len(res) else 0.0
        total_energy = float(np.sum(res[:,4])) if len(res) else 0.0
        print(f"{strat}: ok={ok} | cycles={len(res)} | mean_verify={mean_verify:.2f} | total_energy={total_energy:.2f}")
