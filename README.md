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

## App

```bash
python app/app.py
```

## Safety boundary

CARE-X does not diagnose disease, prescribe treatment, recommend therapy, perform clinical risk scoring, or replace clinical judgment. All included cases are synthetic operational workflow cases.

## License

Apache License 2.0. Copyright (C) 2026 Mohammad Amir Khusru Akhtar.
