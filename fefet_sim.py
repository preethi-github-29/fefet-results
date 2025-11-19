import numpy as np

rng = np.random.default_rng(0)

# --- Parameters (tune these) ---
levels = np.array([0.1, 0.4, 0.7, 1.0])      # target normalized levels (L0..L3)
sigma_device = 0.02                           # device variability
k_stress, alpha, beta = 2e-4, 1.3, 0.8        # trap accumulation model
tau_relax = 5.0                               # rest-time relaxation constant
verify_tol = 0.03                             # verify tolerance
Ceff = 1.0                                    # effective cap for energy estimation (arb.)
margin = 0.25                                 # 25% of nominal spacing as failure criterion

def f_program(vpulse, width):
    """Map a pulse to a 'nudge' in level (smooth saturating)."""
    s = 1.0 / (1.0 + np.exp(-0.8*(vpulse*width - 1.0)))
    return 0.15 * s  # per-pulse increment scale

def update_trap(dv_trap, vpulse, width, t_rest):
    dv_trap = dv_trap + k_stress * (vpulse**alpha) * (width**beta)
    dv_trap = dv_trap * np.exp(-t_rest / tau_relax)
    return dv_trap

def write_with_verify(G_init, target, strategy="S3", max_iter=50, vpulse=2.0, width=1.0, t_rest=0.5):
    """Return final level, trap shift, verify count, energy."""
    G = G_init
    dv_trap = 0.0
    verifies = 0
    energy = 0.0
    for i in range(max_iter):
        if strategy == "S1":  # one-shot
            dG = f_program(vpulse, width)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += (vpulse**2) * width * Ceff
            dv_trap = update_trap(dv_trap, vpulse, width, 0.0)
            verifies += 1
            break
        elif strategy == "S2":  # fixed n pulses
            dG = f_program(vpulse*0.7, width*0.5)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += ((vpulse*0.7)**2) * (width*0.5) * Ceff
            dv_trap = update_trap(dv_trap, vpulse*0.7, width*0.5, 0.0)
        else:  # S3: verify + adjust + rest
            err = target - G
            if abs(err) <= verify_tol:
                break
            v_eff = 1.2 if err > 0 else 0.9
            w_eff = 0.6
            dG = f_program(v_eff, w_eff) * np.sign(err)
            G = np.clip(G + dG - dv_trap, 0, 1)
            energy += (v_eff**2) * w_eff * Ceff
            dv_trap = update_trap(dv_trap, v_eff, w_eff, t_rest)
        verifies += 1
    return G, dv_trap, verifies, energy

def level_spacing_ok(G_levels):
    diffs = np.diff(np.sort(G_levels))
    nominal = np.mean(diffs)
    return np.all(diffs >= margin*nominal)

def simulate_cycles(n_cycles=1000, strategy="S3"):
    G = 0.0
    results = []
    for c in range(n_cycles):
        # cycle through targets L0..L3
        tgt = levels[c % len(levels)] + rng.normal(0, sigma_device)
        G, dv_trap, ver, E = write_with_verify(G, tgt, strategy=strategy)
        results.append((c, G, dv_trap, ver, E))
        # check failure every 50 cycles
        if (c+1) % 50 == 0:
            probe = []
            G_probe = G
            for t in levels:
                G_probe, _, _, _ = write_with_verify(G_probe, t, strategy=strategy, max_iter=10)
                probe.append(G_probe)
            if not level_spacing_ok(np.array(probe)):
                return np.array(results), False
    return np.array(results), True

if __name__ == "__main__":
    res, ok = simulate_cycles(1000, strategy="S3")
    print("Endurance passed:", ok, "| cycles simulated:", len(res))
