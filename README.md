# CARE-X

**Capability-Aware Reliable and Explainable AI for Healthcare Operations**

CARE-X is a reproducible research platform for auditing *why* an augmented healthcare workflow becomes capable. It separates four intervention channels: **R**esources, **I**nformation, **A**ctions/interfaces, and **L**ules, and evaluates all 16 factorial combinations. The project is deliberately **non-diagnostic** and focuses on assistive cancer follow-up/navigation workflows.

## What is distinctive

- **Capability X-Ray:** exact whole-envelope interaction attribution.
- **Failure Simulator:** removes channels and measures safe capability loss.
- **Dependency Map:** minimal enabling sets and minimal failure cuts.
- **Capability Survival Score:** fraction of tested augmentation states under which a workflow remains safely feasible.
- **Dependency Concentration:** normalized concentration of attribution mass.
- **Capability Passport:** per-workflow audit certificate.
- **AUTO / CONFIRM / ESCALATE:** explicit human-oversight states.

## Reproduce

```bash
pip install -r requirements.txt
./reproduce.sh
```

Generated outputs appear under `results/`.

## Reproduced synthetic benchmark

The reference simulator run uses 2,048 seeded synthetic, non-diagnostic workflow cases. It produces exact factorial reconstruction error **0**, higher-order absolute interaction mass **0.72265625**, third-order held-out full-system prediction error **0.02783203125**, and an unsafe-attempt rate **0.27978515625** in the RIA condition when the rules channel is absent. Removing R, I, A, and L from the fully augmented condition causes safe-completion losses of **0.1552734375**, **0.72265625**, **0.412109375**, and **0.81494140625**, respectively. These are properties of the synthetic benchmark and are **not clinical-effectiveness estimates**.

## Frozen learned-model audit

CARE-X also includes a second experiment in which a **RandomForestClassifier with 300 trees is trained once and then frozen**. The learned model is evaluated across the same 16 R/I/A/L intervention states on a disjoint 2,048-case synthetic operational test cohort. The full-feature fixed model has task-classification accuracy **0.53760**. Safe completion rises from **0.02197** at baseline to **0.70752** under RIAL; the 95% bootstrap interval for full safe completion is **[0.68846, 0.72706]**. With rules removed (RIA), technical completion remains **0.53760** but safe completion falls to **0.22363** and unsafe attempts rise to **0.31396** (95% bootstrap interval **[0.29393, 0.33545]**). Exact Möbius reconstruction error is **0**, higher-order absolute interaction mass is **0.20801**, and third-order held-out full-system prediction error is **0.00684**. Across ten independent 1,024-case test cohorts using the same frozen model, mean full safe completion is **0.73223 ± 0.01708** and mean unsafe-attempt rate without rules is **0.33369 ± 0.01554**.

This learned-model experiment is a real fitted-model audit on synthetic non-diagnostic workflow data. It is **not** a clinical validation and is **not** presented as an external pretrained LLM study. Generated CSVs and figures are in `results/real_model/` and the executable experiment is `experiments/run_learned_model.py`.

## App

```bash
python app/app.py
```

## Safety boundary

CARE-X does not diagnose disease, prescribe treatment, recommend therapy, perform clinical risk scoring, or replace clinical judgment. All included cases are synthetic operational workflow cases.

## License

Apache License 2.0. Copyright (C) 2026 Mohammad Amir Khusru Akhtar.
