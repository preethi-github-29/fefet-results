# FeFET Endurance & Programming Strategies

Evaluating endurance-aware programming methods for Ferroelectric Field-Effect Transistors (FeFETs) to enable stable multi-level analog storage.

---

## 🎯 Objective
FeFETs provide fast, low-power, CMOS-compatible multi-level storage for AI accelerators, but face endurance limits ($\sim 10^5–10^6$ writes) due to charge trapping and polarization fatigue. This project evaluates programming strategies to maintain 4 distinct states ($L_0–L_3$) for $\ge 1000$ cycles with high energy efficiency.

---

## ⚙️ Evaluated Strategies
* **S1 (Single Strong Pulse):** Simple and fast, but induces high electrical stress.
* **S2 (Blind Multi-Pulse):** Stable across cycles, but suffers from high energy consumption.
* **S3 (Proposed — Pulse + Verify + Rest):** Combines gentle incremental pulses, verify-and-stop logic, and pause times for trap relaxation.

---

## 📊 Results Summary

| Metric | S1 | S2 | S3 (Proposed) |
| :--- | :--- | :--- | :--- |
| **Cycles Survived** | 1000 | 1000 | **1000** |
| **Mean Verify Count** | 1 | 50 | **16.39** |
| **Total Energy** | ~4,000 | ~49,000 | **~5,057** |
| **Efficiency** | Good | Poor | **Best Trade-off** |

---

## 🔬 Why S3 Works & Real-World Feasibility
* **Trap Mitigation:** Rest intervals allow exponential trap relaxation, reducing cumulative fatigue.
* **Lab Compatibility:** Parameters map directly to standard laboratory equipment (arbitrary waveform generators, parameter analyzers) using ISPP-style programming.
