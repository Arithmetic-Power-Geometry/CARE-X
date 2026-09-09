import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('rma',Path(__file__).resolve().parents[1]/'experiments'/'run_learned_model.py')
rma=importlib.util.module_from_spec(spec); spec.loader.exec_module(rma)

def test_frozen_model_factorial_reconstruction():
    train=rma.generate(1500,11); test=rma.generate(256,12); model=rma.build_model(train,n_estimators=40)
    rows=[rma.evaluate_condition(model,test,S) for S in rma.powerset()]
    vals={S:next(x['safe_completion_rate'] for x in rows if x['condition']==rma.key(S)) for S in rma.powerset()}
    coeff=rma.mobius(vals)
    assert max(abs(sum(v for J,v in coeff.items() if J.issubset(S))-vals[S]) for S in rma.powerset()) < 1e-12

def test_rules_gate_unsafe_attempts():
    train=rma.generate(1500,21); test=rma.generate(256,22); model=rma.build_model(train,n_estimators=40)
    ria=rma.evaluate_condition(model,test,frozenset({'R','I','A'}))
    rial=rma.evaluate_condition(model,test,frozenset({'R','I','A','L'}))
    assert rial['unsafe_attempt_rate']==0.0
    assert ria['unsafe_attempt_rate'] >= rial['unsafe_attempt_rate']
