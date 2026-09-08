from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
import pandas as pd
from carex.simulator import generate_cases, passport

ROOT = Path(__file__).resolve().parents[1]
FACT = ROOT / "results/csv/factorial_results.csv"
FAIL = ROOT / "results/csv/failure_map.csv"
CASES = generate_cases(64, 2026)


def load_tables():
    f = pd.read_csv(FACT) if FACT.exists() else pd.DataFrame()
    m = pd.read_csv(FAIL) if FAIL.exists() else pd.DataFrame()
    return f, m


def make_passport(case_id):
    case = next(c for c in CASES if c.case_id == case_id)
    return json.dumps(passport(case).to_dict(), indent=2)


with gr.Blocks(title="CARE-X") as demo:
    gr.Markdown("# CARE-X\n### Capability-Aware Reliable and Explainable AI for Healthcare Operations\n**Non-diagnostic assistive workflow demonstrator.**")
    with gr.Tab("Capability X-Ray"):
        f, _ = load_tables()
        gr.Dataframe(value=f, interactive=False)
    with gr.Tab("Failure Simulator"):
        _, m = load_tables()
        gr.Dataframe(value=m, interactive=False)
    with gr.Tab("Capability Passport"):
        sel = gr.Dropdown([c.case_id for c in CASES], value=CASES[0].case_id, label="Synthetic case")
        out = gr.Code(language="json", value=make_passport(CASES[0].case_id))
        sel.change(make_passport, sel, out)
    with gr.Tab("Safety Boundary"):
        gr.Markdown("CARE-X does **not** diagnose, prescribe, recommend treatment, or autonomously make clinical decisions. AUTO/CONFIRM/ESCALATE applies only to administrative and coordination workflows.")


if __name__ == "__main__":
    demo.launch()
