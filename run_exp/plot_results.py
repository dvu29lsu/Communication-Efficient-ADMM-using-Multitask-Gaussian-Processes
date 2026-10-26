import os
import pandas as pd
import matplotlib.pyplot as plt

SET_NAME = "Set1"  # change if needed

VANILLA_CSV = os.path.join("results", "vanilla", SET_NAME, "iter_metrics.csv")
SOGP_CSV    = os.path.join("results", "sogp",    SET_NAME, "iter_metrics.csv")
AGENT_CSV   = os.path.join("results", "sogp",    SET_NAME, "agent_metrics.csv")  # optional

OUT_DIR = os.path.join("results", "plots", SET_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)

van = load_df(VANILLA_CSV)
sogp = load_df(SOGP_CSV)

# Align length (in case one ends earlier)
K = min(len(van), len(sogp))
van = van.iloc[:K].copy()
sogp = sogp.iloc[:K].copy()
k = van["iter"].values

# -----------------------------
# 1) Communication per iteration
# -----------------------------
plt.figure()
plt.plot(k, van["q"].values, label="Vanilla (full comm)")
plt.plot(k, sogp["q"].values, label="SOGP (variance gate)")
plt.xlabel("Iteration k")
plt.ylabel("Comm per iter (# queried agents)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comm_per_iter.png"), dpi=200)

# -----------------------------
# 2) Cumulative communication
# -----------------------------
# If comm_cum exists use it, else build it
van_cum = van["comm_cum"].values if "comm_cum" in van.columns else van["q"].cumsum().values
sogp_cum = sogp["comm_cum"].values if "comm_cum" in sogp.columns else sogp["q"].cumsum().values

plt.figure()
plt.plot(k, van_cum, label="Vanilla cumulative comm")
plt.plot(k, sogp_cum, label="SOGP cumulative comm")
plt.xlabel("Iteration k")
plt.ylabel("Cumulative comm (total queries)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comm_cumulative.png"), dpi=200)

# Print comm reduction summary
van_total = float(van_cum[-1])
sogp_total = float(sogp_cum[-1])
print("Total comm vanilla:", int(van_total))
print("Total comm SOGP:", int(sogp_total))
if van_total > 0:
    print("Reduction:", f"{100.0*(1.0 - sogp_total/van_total):.2f}%")

# -----------------------------
# 3) Primal/Dual residuals (log scale)
# -----------------------------
plt.figure()
plt.semilogy(k, van["r"].values, label="Vanilla r (primal)")
plt.semilogy(k, sogp["r"].values, label="SOGP r (primal)")
plt.xlabel("Iteration k")
plt.ylabel("Primal residual r_k")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "primal_residual.png"), dpi=200)

plt.figure()
plt.semilogy(k, van["s"].values, label="Vanilla s (dual)")
plt.semilogy(k, sogp["s"].values, label="SOGP s (dual)")
plt.xlabel("Iteration k")
plt.ylabel("Dual residual s_k")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dual_residual.png"), dpi=200)

# -----------------------------
# 4) Objective curve
# -----------------------------
if "obj" in van.columns and "obj" in sogp.columns:
    plt.figure()
    plt.plot(k, van["obj"].values, label="Vanilla objective")
    plt.plot(k, sogp["obj"].values, label="SOGP objective")
    plt.xlabel("Iteration k")
    plt.ylabel("Objective value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "objective.png"), dpi=200)

# -----------------------------
# 5) (Optional) Show gate behavior: stdmax vs psi (SOGP)
# -----------------------------
# Only if these columns exist (they do in your SOGP iter CSV)
if "stdmax_max" in sogp.columns and "psi_mean" in sogp.columns:
    plt.figure()
    plt.plot(k, sogp["stdmax_max"].values, label="max std (over agents)")
    plt.plot(k, sogp["psi_mean"].values, label="mean psi (threshold)")
    plt.xlabel("Iteration k")
    plt.ylabel("Uncertainty / Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "uncertainty_vs_threshold.png"), dpi=200)

print("Saved plots to:", OUT_DIR)

# Optional agent-level heatmap-like view (requires seaborn, so skipped)
# But you can quickly inspect who queried when using pivot tables.
if os.path.exists(AGENT_CSV):
    agent = pd.read_csv(AGENT_CSV)
    # quick summary:
    print("Avg queried agents per iter (from agent CSV):",
          agent.groupby("iter")["did_query"].sum().mean())