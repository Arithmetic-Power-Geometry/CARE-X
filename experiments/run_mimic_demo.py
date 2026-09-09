from __future__ import annotations

import argparse, json, urllib.request
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "https://physionet.org/files/mimic-iv-ed-demo/2.2/ed/"
FILES = ["edstays.csv.gz", "triage.csv.gz", "vitalsign.csv.gz", "medrecon.csv.gz", "pyxis.csv.gz"]
CHANNELS = ("R", "I", "A", "L")


def powerset(items=CHANNELS):
    return [frozenset(c) for r in range(len(items)+1) for c in combinations(items, r)]


def label(s):
    return "BASE" if not s else "".join(ch for ch in CHANNELS if ch in s)


def download(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for fn in FILES:
        p = data_dir / fn
        if not p.exists():
            urllib.request.urlretrieve(BASE + fn, p)


def load_tables(data_dir: Path):
    return {fn.split('.')[0]: pd.read_csv(data_dir/fn, compression='gzip') for fn in FILES}


def build_stays(t):
    stays = t['edstays'].copy()
    tri = t['triage'].copy()
    tri['triage_present'] = tri['chiefcomplaint'].fillna('').astype(str).str.strip().ne('')
    tri2 = tri.groupby('stay_id', as_index=False).agg(triage_present=('triage_present','max'))
    stays = stays.merge(tri2, on='stay_id', how='left')
    stays['triage_present'] = stays['triage_present'].fillna(False).astype(bool)
    for src in ['vitalsign','medrecon','pyxis']:
        c = t[src].groupby('stay_id').size().rename(src+'_n')
        stays = stays.merge(c, on='stay_id', how='left')
        stays[src+'_n'] = stays[src+'_n'].fillna(0).astype(int)
    stays['event_n'] = stays[['vitalsign_n','medrecon_n','pyxis_n']].sum(axis=1)
    med = float(stays['event_n'].median())
    stays['record_heavy'] = stays['event_n'] > med
    stays['disposition_present'] = stays['disposition'].fillna('').astype(str).str.strip().ne('')
    return stays, med


def cases_from_stays(stays):
    rows=[]
    for r in stays.itertuples(index=False):
        base=dict(subject_id=int(r.subject_id),stay_id=int(r.stay_id),gender=str(r.gender),arrival_transport=str(r.arrival_transport),
                  triage_present=bool(r.triage_present),vitalsign_n=int(r.vitalsign_n),medrecon_n=int(r.medrecon_n),
                  pyxis_n=int(r.pyxis_n),record_heavy=bool(r.record_heavy),disposition_present=bool(r.disposition_present))
        for task in ['intake_document_packet','medication_document_reconciliation','disposition_handoff']:
            rows.append({**base,'task':task})
    return pd.DataFrame(rows)


def eval_case(r, S, no_resource_gate=False):
    R,I,A,L = (x in S for x in CHANNELS)
    resource_ok = True if no_resource_gate else ((not bool(r.record_heavy)) or R)
    if r.task == 'intake_document_packet':
        technical = bool(I and r.triage_present and resource_ok)
        safe = technical
        escalation = False
        unsafe = False
    elif r.task == 'medication_document_reconciliation':
        technical = bool(I and A and (r.medrecon_n > 0) and resource_ok)
        declared_prereq = bool((r.medrecon_n > 0) and ((r.pyxis_n > 0) or (r.vitalsign_n > 0)))
        unsafe_raw = bool(technical and not declared_prereq)
        escalation = bool(L and unsafe_raw)
        unsafe = bool(unsafe_raw and not L)
        safe = bool(technical and declared_prereq) or escalation
    else:
        technical = bool(A and r.disposition_present and resource_ok)
        declared_prereq = bool(I and r.triage_present)
        unsafe_raw = bool(technical and not declared_prereq)
        escalation = bool(L and unsafe_raw)
        unsafe = bool(unsafe_raw and not L)
        safe = bool(technical and declared_prereq) or escalation
    return technical, safe, escalation, unsafe


def profile(cases, no_resource_gate=False):
    out=[]
    point=[]
    for S in powerset():
        vals=np.array([eval_case(r,S,no_resource_gate) for r in cases.itertuples(index=False)], dtype=float)
        out.append(dict(condition=label(S), completion=vals[:,0].mean(), safe_completion=vals[:,1].mean(),
                        escalation=vals[:,2].mean(), unsafe_attempt=vals[:,3].mean()))
        for i,v in enumerate(vals[:,1]): point.append((i,label(S),v))
    return pd.DataFrame(out), pd.DataFrame(point,columns=['case_id','condition','safe'])


def mobius_from_rates(df):
    v={frozenset(x for x in CHANNELS if x in c):float(y) for c,y in zip(df.condition,df.safe_completion)}
    # BASE needs explicit empty set handling
    v[frozenset()] = float(df.loc[df.condition=='BASE','safe_completion'].iloc[0])
    d={}
    for J in powerset():
        d[J]=sum(((-1)**(len(J)-len(K)))*v[K] for K in powerset(J))
    rec={S:sum(d[J] for J in d if J.issubset(S)) for S in v}
    maxerr=max(abs(rec[S]-v[S]) for S in v)
    mass={q:sum(abs(z) for J,z in d.items() if len(J)==q) for q in range(5)}
    held=[]
    full=frozenset(CHANNELS)
    for q in range(5):
        pred=sum(z for J,z in d.items() if len(J)<=q and J.issubset(full))
        held.append(dict(max_order=q,prediction=pred,actual=v[full],absolute_error=abs(pred-v[full])))
    return d,maxerr,mass,pd.DataFrame(held)


def failure_map(df):
    m={c:r for c,r in df.set_index('condition').iterrows()}
    full=float(m['RIAL'].safe_completion)
    rows=[]
    for ch,cond in [('R','IAL'),('I','RAL'),('A','RIL'),('L','RIA')]:
        val=float(m[cond].safe_completion)
        rows.append(dict(removed_channel=ch,safe_completion=val,loss_from_full=full-val))
    return pd.DataFrame(rows)


def patient_bootstrap(cases, B=2000, seed=20260909):
    rng=np.random.default_rng(seed); ids=cases.subject_id.unique(); stats=[]
    targets=['BASE','RIA','RIAL']
    for b in range(B):
        samp=rng.choice(ids,size=len(ids),replace=True)
        # preserve multiplicity by concatenating patient blocks
        cb=pd.concat([cases[cases.subject_id==x] for x in samp],ignore_index=True)
        p,_=profile(cb)
        row={'replicate':b}
        for c in targets:
            rr=p[p.condition==c].iloc[0]
            row[c+'_safe']=rr.safe_completion; row[c+'_unsafe']=rr.unsafe_attempt
        stats.append(row)
    boot=pd.DataFrame(stats)
    ci=[]
    for col in [c for c in boot.columns if c!='replicate']:
        ci.append(dict(metric=col,estimate=float(boot[col].mean()),lo=float(boot[col].quantile(.025)),hi=float(boot[col].quantile(.975))))
    return boot,pd.DataFrame(ci)


def subgroup(cases):
    rows=[]
    for field in ['gender','arrival_transport']:
        for val,g in cases.groupby(field,dropna=False):
            p,_=profile(g)
            for c in ['RIA','RIAL']:
                r=p[p.condition==c].iloc[0]
                rows.append(dict(subgroup=field,value=str(val),n_cases=len(g),condition=c,safe_completion=r.safe_completion,unsafe_attempt=r.unsafe_attempt))
    return pd.DataFrame(rows)


def dependency_prediction(cases):
    ids=np.array(sorted(cases.subject_id.unique()))
    # deterministic patient split; no stay from one patient crosses split
    cut=max(1,int(.6*len(ids))); disc=set(ids[:cut]); test=set(ids[cut:])
    rows=[]
    best={}
    for name,subset in [('discovery',disc),('test',test)]:
        p,_=profile(cases[cases.subject_id.isin(subset)])
        fm=failure_map(p)
        top=fm.sort_values(['loss_from_full','removed_channel'],ascending=[False,True]).iloc[0]
        best[name]=str(top.removed_channel)
        for r in fm.itertuples(index=False): rows.append(dict(split=name,removed_channel=r.removed_channel,loss_from_full=r.loss_from_full))
    return pd.DataFrame(rows), dict(discovery_top=best['discovery'],test_top=best['test'],top_dependency_replicates=(best['discovery']==best['test']))


def main(outdir: Path, data_dir: Path):
    download(data_dir); t=load_tables(data_dir); stays,median_events=build_stays(t); cases=cases_from_stays(stays)
    outdir.mkdir(parents=True,exist_ok=True); csv=outdir/'csv'; fig=outdir/'figures'; csv.mkdir(exist_ok=True); fig.mkdir(exist_ok=True)
    prof,_=profile(cases); d,maxerr,mass,held=mobius_from_rates(prof); fm=failure_map(prof)
    boot,ci=patient_bootstrap(cases); sub=subgroup(cases); dep,dep_summary=dependency_prediction(cases)
    ab,_=profile(cases,no_resource_gate=True)
    prof.to_csv(csv/'factorial_mimic_demo.csv',index=False); fm.to_csv(csv/'failure_map_mimic_demo.csv',index=False)
    held.to_csv(csv/'heldout_mimic_demo.csv',index=False); pd.DataFrame([{'order':k,'absolute_mass':v} for k,v in mass.items()]).to_csv(csv/'interaction_mass_mimic_demo.csv',index=False)
    pd.DataFrame([{'subset':label(J),'coefficient':z} for J,z in d.items()]).to_csv(csv/'mobius_mimic_demo.csv',index=False)
    ci.to_csv(csv/'bootstrap_ci_mimic_demo.csv',index=False); sub.to_csv(csv/'subgroups_mimic_demo.csv',index=False); dep.to_csv(csv/'dependency_prediction_mimic_demo.csv',index=False); ab.to_csv(csv/'ablation_no_resource_gate.csv',index=False)
    profile_rows={k:len(v) for k,v in t.items()}; profile_rows.update({'unique_patients':int(stays.subject_id.nunique()),'unique_stays':int(stays.stay_id.nunique()),'workflow_cases':int(len(cases)),'median_event_count':median_events})
    pd.DataFrame([profile_rows]).to_csv(csv/'dataset_profile_mimic_demo.csv',index=False)
    leakage={'diagnosis_table_loaded':False,'learned_target':None,'uses_patient_identifier_as_feature':False,'uses_disposition_as_prediction_label':False,'future_outcome_prediction':False,'patient_level_split_for_dependency_test':True}
    (outdir/'leakage_audit.json').write_text(json.dumps(leakage,indent=2))
    full=float(prof.loc[prof.condition=='RIAL','safe_completion'].iloc[0]); ria=prof.loc[prof.condition=='RIA'].iloc[0]
    summary={'source':'PhysioNet MIMIC-IV-ED Demo v2.2','scope':'pipeline verification; non-diagnostic; not clinical validation','dataset':profile_rows,'exact_reconstruction_error':maxerr,'full_safe_completion':full,'ria_safe_completion':float(ria.safe_completion),'ria_technical_completion':float(ria.completion),'ria_unsafe_attempt':float(ria.unsafe_attempt),'higher_order_abs_mass':sum(v for k,v in mass.items() if k>=2),'order3_heldout_error':float(held.loc[held.max_order==3,'absolute_error'].iloc[0]),'dependency_prediction':dep_summary}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2))
    x=np.arange(len(prof)); plt.figure(figsize=(9,4)); plt.bar(x,prof.safe_completion); plt.xticks(x,prof.condition,rotation=60); plt.ylabel('Safe workflow handling rate'); plt.xlabel('Enabled channels'); plt.tight_layout(); plt.savefig(fig/'factorial_mimic_demo.svg'); plt.close()
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,default=Path('results/mimic_demo')); ap.add_argument('--data-dir',type=Path,default=Path('data/mimic_iv_ed_demo_2_2')); a=ap.parse_args(); main(a.outdir,a.data_dir)
