from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import combinations
from math import log
from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple

CHANNELS: Tuple[str, ...] = ("R", "I", "A", "L")

def powerset(items: Sequence[str] = CHANNELS) -> List[FrozenSet[str]]:
    out=[]
    for r in range(len(items)+1): out.extend(frozenset(c) for c in combinations(items,r))
    return out

def mobius(values: Mapping[FrozenSet[str], float]) -> Dict[FrozenSet[str], float]:
    coeff={}
    for J in powerset():
        total=0.0; jlist=tuple(sorted(J))
        for K in powerset(jlist): total += ((-1)**(len(J)-len(K)))*float(values.get(K,0.0))
        coeff[J]=total
    return coeff

def reconstruct(coeff, S): return sum(v for J,v in coeff.items() if J.issubset(S))
def truncated_predict(coeff,S,order): return sum(v for J,v in coeff.items() if J.issubset(S) and len(J)<=order)
def interaction_mass(coeff): return {r:sum(abs(v) for J,v in coeff.items() if len(J)==r) for r in range(5)}
def capability_survival_score(feasible,exclude_full=False):
    states=powerset(); states=[s for s in states if len(s)<len(CHANNELS)] if exclude_full else states
    return sum(bool(feasible.get(s,False)) for s in states)/len(states)
def weighted_survival_score(feasible,p_available):
    score=0.0
    for S in powerset():
        prob=1.0
        for ch in CHANNELS:
            p=float(p_available[ch]); prob*=p if ch in S else (1-p)
        score += prob*float(bool(feasible.get(S,False)))
    return score
def dependency_concentration(coeff,eps=1e-12):
    masses=[abs(v) for J,v in coeff.items() if J and abs(v)>eps]
    if len(masses)<=1: return 1.0 if masses else 0.0
    total=sum(masses); p=[m/total for m in masses]; H=-sum(x*log(x) for x in p)
    return 1.0-H/log(len(p))
def minimal_enabling_sets(feasible):
    mins=[]
    for S in sorted(powerset(),key=lambda x:(len(x),tuple(sorted(x)))):
        if feasible.get(S,False) and not any(T<S and feasible.get(T,False) for T in powerset(tuple(S))): mins.append(S)
    return mins
def minimal_failure_cuts(feasible):
    full=frozenset(CHANNELS); cuts=[]
    for removed in sorted(powerset(),key=lambda x:(len(x),tuple(sorted(x)))):
        if not feasible.get(full-removed,False):
            if not any(C<removed and not feasible.get(full-C,False) for C in powerset(tuple(removed))): cuts.append(removed)
    return cuts
@dataclass(frozen=True)
class Case:
    case_id:str; task:str; requirements:FrozenSet[str]; language:str; caregiver:bool; ambiguous:bool; missing_info:bool; rule_sensitive:bool
@dataclass
class Passport:
    case_id:str; task:str; full_safe_feasible:bool; capability_survival_score:float; dependency_concentration:float; minimal_enabling_sets:List[List[str]]; minimal_failure_cuts:List[List[str]]; human_override:str; diagnostic_scope:str="NON-DIAGNOSTIC ASSISTIVE WORKFLOW ONLY"
    def to_dict(self): return asdict(self)
