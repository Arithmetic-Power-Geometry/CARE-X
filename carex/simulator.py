from __future__ import annotations
import random
from collections import defaultdict
from typing import FrozenSet
from .core import CHANNELS,Case,Passport,capability_survival_score,dependency_concentration,minimal_enabling_sets,minimal_failure_cuts,mobius,powerset
TASK_REQUIREMENTS={"reminder_generation":frozenset({"I","L"}),"appointment_reschedule":frozenset({"I","A","L"}),"missing_information_escalation":frozenset({"L"}),"caregiver_coordination":frozenset({"I","A","L"}),"multilingual_message":frozenset({"I","A"}),"care_team_handoff":frozenset({"I","L"}),"document_reconciliation":frozenset({"R","I"}),"status_tracking":frozenset({"I","A"})}
def generate_cases(n=512,seed=2026):
    rng=random.Random(seed); tasks=list(TASK_REQUIREMENTS); langs=["English","Hindi","Bengali","Odia"]; cases=[]
    for i in range(n):
        task=tasks[i%len(tasks)] if i<len(tasks) else rng.choice(tasks); req=set(TASK_REQUIREMENTS[task]); ambiguous=rng.random()<0.28; missing_info=rng.random()<0.18; rule_sensitive=rng.random()<0.34; caregiver=rng.random()<0.42; lang=rng.choice(langs)
        if ambiguous and task in {"document_reconciliation","care_team_handoff","appointment_reschedule"}: req.add("R")
        if lang!="English" and task in {"multilingual_message","caregiver_coordination"}: req.add("A")
        cases.append(Case(f"CX-{i+1:04d}",task,frozenset(req),lang,caregiver,ambiguous,missing_info,rule_sensitive))
    return cases
def evaluate_case(case,enabled:FrozenSet[str]):
    if case.missing_info:
        if "L" in enabled: return True,True,"ESCALATE"
        completion=case.requirements-{"L"}<=enabled
        return completion,False if completion else True,"UNSAFE_ATTEMPT" if completion else "BLOCKED"
    completion=case.requirements<=enabled
    if not completion: return False,True,"BLOCKED"
    if case.rule_sensitive and "L" not in enabled: return True,False,"UNSAFE_ATTEMPT"
    if case.task in {"appointment_reschedule","caregiver_coordination"}: return True,True,"CONFIRM"
    return True,True,"AUTO"
def benchmark(cases):
    rows=[]; cases=list(cases)
    for S in powerset():
        comp=safe=escal=unsafe=0
        for c in cases:
            completed,compliant,action=evaluate_case(c,S); comp+=int(completed); safe+=int(completed and compliant); escal+=int(action=="ESCALATE"); unsafe+=int(action=="UNSAFE_ATTEMPT")
        rows.append({"condition":"".join(ch for ch in CHANNELS if ch in S) or "BASE","order":len(S),"completion_rate":comp/len(cases),"safe_completion_rate":safe/len(cases),"escalation_rate":escal/len(cases),"unsafe_action_rate":unsafe/len(cases)})
    return rows
def case_feasibility(case): return {S:bool(evaluate_case(case,S)[0] and evaluate_case(case,S)[1]) for S in powerset()}
def passport(case):
    feas=case_feasibility(case); coeff=mobius({S:float(v) for S,v in feas.items()})
    return Passport(case.case_id,case.task,feas[frozenset(CHANNELS)],capability_survival_score(feas),dependency_concentration(coeff),[sorted(x) for x in minimal_enabling_sets(feas)],[sorted(x) for x in minimal_failure_cuts(feas)],"Enabled (AUTO / CONFIRM / ESCALATE)")
def task_level_attribution(cases):
    grouped=defaultdict(list)
    for c in cases: grouped[c.task].append(c)
    out={}
    for task,group in grouped.items():
        vals={S:sum(evaluate_case(c,S)[0] and evaluate_case(c,S)[1] for c in group)/len(group) for S in powerset()}; out[task]=mobius(vals)
    return out
