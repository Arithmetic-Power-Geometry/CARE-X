from carex.core import *
from carex.simulator import generate_cases,benchmark,case_feasibility
def test_mobius_exact_reconstruction():
    vals={S:float(len(S)>=2) for S in powerset()}; c=mobius(vals); assert max(abs(reconstruct(c,S)-vals[S]) for S in powerset())<1e-12
def test_css_range():
    c=generate_cases(1)[0]; x=capability_survival_score(case_feasibility(c)); assert 0<=x<=1
def test_full_has_safe_capability():
    rows={r['condition']:r for r in benchmark(generate_cases(64))}; assert rows['RIAL']['safe_completion_rate']>=rows['BASE']['safe_completion_rate']
def test_rules_reduce_unsafe_attempts():
    rows={r['condition']:r for r in benchmark(generate_cases(128))}; assert rows['RIAL']['unsafe_action_rate']==0
def test_minimal_sets_are_minimal():
    c=generate_cases(1)[0]; f=case_feasibility(c)
    for S in minimal_enabling_sets(f): assert f[S] and all(not f[T] for T in powerset(tuple(S)) if T<S)
def test_weighted_survival_bounds():
    c=generate_cases(1)[0]; s=weighted_survival_score(case_feasibility(c),{ch:0.9 for ch in CHANNELS}); assert 0<=s<=1
