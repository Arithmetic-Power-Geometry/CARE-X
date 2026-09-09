# CARE-X Phase 1: MIMIC-IV-ED Demo protocol

## Status
This protocol is frozen before inspection of CARE-X factorial results from the real-data demo run. It is a pipeline-verification study, not clinical validation.

## Data
Primary source: PhysioNet MIMIC-IV-ED Demo v2.2 (100 de-identified patients). The script downloads the official `edstays`, `triage`, `vitalsign`, `medrecon`, and `pyxis` tables directly from PhysioNet. The `diagnosis` table is deliberately excluded so that the experiment remains non-diagnostic.

## Unit of analysis
Each ED stay is expanded into three non-diagnostic workflow-audit tasks: intake-document packet, medication-document reconciliation, and disposition handoff. The experiment uses real record-presence/event-count patterns but applies a declared synthetic audit policy. Therefore an `unsafe_attempt` means violation of the declared experimental workflow rule, not demonstrated clinical harm.

## Intervention channels
- **R — Resources:** extended reconciliation/processing budget for record-heavy stays.
- **I — Information:** access to the relevant structured documentation source.
- **A — Actions/interfaces:** access to downstream structured event/interface records.
- **L — Rules:** explicit rule gate that blocks or escalates when declared prerequisites are not met.

All 16 subsets of `{R,I,A,L}` are evaluated without changing the underlying stay.

## Pre-specified task logic
A stay is `record_heavy` when its total counted vitalsign, medrecon, and pyxis events exceed the median of the demo cohort.

1. **intake_document_packet**: technical output requires I, a triage record with a non-empty chief complaint, and R when the stay is record-heavy. It is not assigned a rule-gated unsafe-attempt pathway.
2. **medication_document_reconciliation**: technical output requires I and A, at least one medication-reconciliation record, and R when record-heavy. The declared safety prerequisite is the availability of both the medication-reconciliation source and an interface/event source. If a technical attempt occurs without the declared prerequisite, L converts the attempt to ESCALATE; without L it is counted as an unsafe attempt.
3. **disposition_handoff**: technical output requires A and a recorded ED disposition, and R when record-heavy. The declared safety prerequisite additionally requires I and a triage record. L converts a prerequisite failure to ESCALATE; without L it is counted as an unsafe attempt.

Escalation counts as safe workflow handling only for the two pre-specified rule-gated task families.

## Primary analysis
For each intervention subset report technical completion, safe completion, escalation, and unsafe-attempt rates. Apply exact Boolean-lattice Möbius inversion to safe completion. Report exact reconstruction error, interaction mass by order, channel-removal losses from RIAL, and order-0 through order-4 prediction of the held-out RIAL endpoint.

## Robustness analyses
- 2,000 patient-cluster bootstrap replicates for selected endpoint confidence intervals.
- Subgroup description by administrative gender and arrival transport, with no causal or clinical interpretation.
- Dependency prediction: infer the largest single-channel removal loss on a discovery split of patients and test whether that same channel has the largest loss in the held-out patient split.
- Ablation: repeat the primary factorial profile after removing the record-heavy/R resource requirement.
- Leakage audit: verify that no diagnosis table, disposition outcome as a prediction label, future timestamp-derived clinical outcome, or patient identifier is used as a learned target. The experiment is deterministic and performs no diagnostic prediction.

## Decision rule
Phase 1 passes if the code downloads the official demo, joins stay-linked tables without duplicate stay loss, evaluates all 16 intervention states, reconstructs the factorial endpoint to numerical precision, emits all pre-specified robustness outputs, and clearly preserves the non-diagnostic scope. Numerical effect sizes are reported as observed and are not success criteria.

## Phase 2 freeze
The code paths, channel definitions, task logic, endpoints, and robustness analyses are to be frozen before applying the pipeline to credentialed full MIMIC-IV-ED/MIMICEL data. Any later change must be versioned and justified rather than silently tuned to improve results.
