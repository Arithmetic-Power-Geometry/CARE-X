from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from .core import CHANNELS, mobius, powerset, interaction_mass, truncated_predict, reconstruct
from .simulator import benchmark, generate_cases, passport, task_level_attribution


def _key(S):
    return "".join(ch for ch in CHANNELS if ch in S) or "BASE"


def run_all(root: str | Path = ".", n: int = 2048, seed: int = 2026):
    root = Path(root)
    for p in [root/"results/csv", root/"results/figures", root/"results/passports", root/"results/reports", root/"data"]:
        p.mkdir(parents=True, exist_ok=True)

    cases = generate_cases(n=n, seed=seed)
    case_df = pd.DataFrame([{
        "case_id": c.case_id, "task": c.task, "requirements": "".join(sorted(c.requirements)),
        "language": c.language, "caregiver": c.caregiver, "ambiguous": c.ambiguous,
        "missing_info": c.missing_info, "rule_sensitive": c.rule_sensitive
    } for c in cases])
    case_df.to_csv(root/"data/synthetic_workflow_cases.csv", index=False)

    df = pd.DataFrame(benchmark(cases))
    df.to_csv(root/"results/csv/factorial_results.csv", index=False)

    vals = {S: float(df.loc[df.condition == _key(S), "safe_completion_rate"].iloc[0]) for S in powerset()}
    coeff = mobius(vals)
    pd.DataFrame([{"subset": _key(S), "order": len(S), "coefficient": v} for S, v in coeff.items()]).to_csv(root/"results/csv/mobius_coefficients.csv", index=False)

    masses = interaction_mass(coeff)
    pd.DataFrame([{"order": k, "absolute_mass": v} for k, v in masses.items()]).to_csv(root/"results/csv/interaction_mass.csv", index=False)

    full = frozenset(CHANNELS)
    heldout = []
    for q in [0, 1, 2, 3, 4]:
        pred = truncated_predict(coeff, full, q)
        heldout.append({"max_order": q, "prediction": pred, "actual": vals[full], "absolute_error": abs(pred - vals[full])})
    pd.DataFrame(heldout).to_csv(root/"results/csv/heldout_prediction.csv", index=False)

    failure = []
    for ch in CHANNELS:
        rate = vals[full - {ch}]
        failure.append({"removed": ch, "safe_completion_rate": rate, "loss_from_full": vals[full] - rate})
    pd.DataFrame(failure).to_csv(root/"results/csv/failure_map.csv", index=False)

    for c in cases[:12]:
        (root/f"results/passports/{c.case_id}.json").write_text(json.dumps(passport(c).to_dict(), indent=2), encoding="utf-8")

    task_attr = task_level_attribution(cases)
    pd.DataFrame([{"task": task, "subset": _key(S), "order": len(S), "coefficient": v} for task, cs in task_attr.items() for S, v in cs.items()]).to_csv(root/"results/csv/task_attribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df.condition, df.safe_completion_rate)
    ax.set_ylabel("Safe workflow completion rate")
    ax.set_xlabel("Enabled augmentation channels")
    ax.set_title("CARE-X factorial intervention profile")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(root/"results/figures/factorial_profile.png", dpi=220)
    fig.savefig(root/"results/figures/factorial_profile.svg")
    plt.close(fig)

    im = pd.DataFrame([{"order": k, "absolute_mass": v} for k, v in masses.items()])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(im.order.astype(str), im.absolute_mass)
    ax.set_xlabel("Interaction order")
    ax.set_ylabel("Absolute attribution mass")
    ax.set_title("Capability X-Ray interaction mass")
    fig.tight_layout()
    fig.savefig(root/"results/figures/interaction_mass.png", dpi=220)
    fig.savefig(root/"results/figures/interaction_mass.svg")
    plt.close(fig)

    G = nx.DiGraph()
    for task in sorted(case_df.task.unique()):
        G.add_node(task)
        for ch in CHANNELS:
            if case_df.loc[case_df.task == task, "requirements"].str.contains(ch).any():
                G.add_edge(ch, task)
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=2026)
    nx.draw_networkx(G, pos=pos, ax=ax, font_size=8, node_size=1100, arrows=True)
    ax.set_title("CARE-X capability dependency graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(root/"results/figures/dependency_graph.png", dpi=220)
    fig.savefig(root/"results/figures/dependency_graph.svg")
    plt.close(fig)

    stress_rows = []
    for stress_seed in range(2001, 2051):
        sdf = pd.DataFrame(benchmark(generate_cases(512, stress_seed)))
        svals = {S: float(sdf.loc[sdf.condition == _key(S), "safe_completion_rate"].iloc[0]) for S in powerset()}
        scoeff = mobius(svals)
        stress_rows.append({
            "seed": stress_seed,
            "full_safe_completion": svals[full],
            "order3_error": abs(truncated_predict(scoeff, full, 3) - svals[full]),
            "higher_order_abs_mass": sum(v for k, v in interaction_mass(scoeff).items() if k >= 2),
            "unsafe_without_rules": float(sdf.loc[sdf.condition == "RIA", "unsafe_action_rate"].iloc[0]),
        })
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(root/"results/csv/stress_50seeds.csv", index=False)

    summary = {
        "n_cases": len(cases),
        "baseline_safe_completion": vals[frozenset()],
        "full_safe_completion": vals[full],
        "absolute_gain": vals[full] - vals[frozenset()],
        "higher_order_absolute_mass": sum(v for k, v in masses.items() if k >= 2),
        "max_exact_reconstruction_error": max(abs(reconstruct(coeff, S) - vals[S]) for S in powerset()),
        "heldout_order3_error": heldout[3]["absolute_error"],
        "unsafe_action_rate_without_rules": float(df.loc[df.condition == "RIA", "unsafe_action_rate"].iloc[0]),
        "stress_order3_error_mean": float(stress_df.order3_error.mean()),
        "stress_order3_error_sd": float(stress_df.order3_error.std(ddof=1)),
        "stress_unsafe_without_rules_mean": float(stress_df.unsafe_without_rules.mean()),
        "stress_unsafe_without_rules_sd": float(stress_df.unsafe_without_rules.std(ddof=1)),
    }
    (root/"results/reports/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(Path(__file__).resolve().parents[1]), indent=2))
