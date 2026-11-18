"""Read-only reconstruction of Alves (2026)'s released LLM/PPI choice fits.

Source: https://doi.org/10.5281/zenodo.23140919, CC BY 4.0.
The original protocol/code and output archives remain local inputs. This script
does not import or execute the downloaded code, train a predictor, query an LLM,
or generate responses. Design coding, split seeds, probability handling and the
opt-out-derived power follow the attributed original pipeline, lines 102-210,
813-830 and 849-910. Finite-support LPs are classical convex-support checks.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
CORE = "Estudo2_protocolo_e_codigo.zip"
OUTPUT = "saida_confirmatorio_qwen3-30b-a3b-instruct-2507-q4_k_m.zip"
HASHES = {
    CORE: "3d2e5bcd1d3715d4bc8606dccbce81657c2b70a929184f0e6c3992829328e4a5",
    OUTPUT: "7a8a093f1bee280732133aa7e8ad1baf57c07063464f39e55f4d08e7ae73274f",
}
EXCLUDE = {"unique_id", "choice_card", "option", "choice", "index", "choice_set",
           "firstclick", "lastclick", "pagesubmit", "clickcount", "treated"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read_table(core, context):
    """Reconstruct only target utility coding and the protected target segment."""
    book = pd.ExcelFile(io.BytesIO(core.read(context + ".xlsx")))
    people = pd.read_excel(book, "survey")
    choices = pd.read_excel(book, "choices" if "choices" in book.sheet_names else "choice")
    for frame in [people, choices]:
        frame.columns = [{"labl": "label"}.get(str(c).strip(), str(c).strip()) for c in frame.columns]
    people = people[["unique_id", "age"]].drop_duplicates("unique_id").set_index("unique_id").fillna(-99)
    price = next(c for c in choices if any(c.lower().startswith(s) for s in ["cost", "price", "prezzo", "precio"]))
    attrs = [c for c in choices if c not in EXCLUDE and c != price and pd.api.types.is_numeric_dtype(choices[c])]
    offers = choices[choices.option.ne(3)]
    assert choices.loc[choices.option.eq(3), attrs + [price]].abs().sum(axis=1).eq(0).all()
    specs = []
    for attr in attrs:
        levels = sorted(v for v in offers[attr].unique() if v != 0)
        if len(levels) >= 2:
            specs.extend((attr, level) for level in levels[1:])
    names = [f"{a}=={v}" for a, v in specs]
    scale = float(offers[price].max()) or 1.0
    names += [f"{price}/{scale:g}", "ASC_optout"]
    # Original code selects the level-2 label coefficient by the column name.
    idx = names.index("label==2")
    tasks = {}
    for key, frame in choices.groupby(["unique_id", "choice_card"]):
        frame = frame.sort_values("option")
        if len(frame) != 3:
            continue
        assert frame.option.tolist() == [1, 2, 3]
        assert frame.choice.sum() == 1
        x = np.zeros((3, len(names)))
        for j, row in enumerate(frame.itertuples(index=False)):
            values = row._asdict()
            if values["option"] == 3:
                x[j, -1] = 1
            else:
                x[j, :len(specs)] = [float(values[a] == v) for a, v in specs]
                x[j, -2] = values[price] / scale
        tasks[key] = (x, int(np.argmax(frame.choice.to_numpy())))
    ids = sorted({p for p, _ in tasks} & set(people.index))
    return people.loc[ids], tasks, names, idx


def split_target(people, context, rep):
    rng = np.random.default_rng(20261003 + 1000 * rep + sum(map(ord, context)))
    target = rng.permutation(people.index[people.age.isin([1, 2])].to_numpy())
    # The subsequent nonsegment permutation does not change HT/VT.
    return set(target[:35]), set(target[35:50])


def subset(tasks, predictions, people):
    keys = sorted(k for k in tasks if k[0] in people)
    frame = predictions.loc[keys]
    assert not frame.index.duplicated().any()
    valid = (frame.n_validas >= .5 * (frame.n_validas + frame.n_invalidas)).to_numpy()
    valid &= np.isfinite(frame[["p1", "p2", "p3"]].to_numpy(float)).all(axis=1)
    kept = [k for k, ok in zip(keys, valid) if ok]
    frame = frame.loc[valid]
    assert frame.prob_nativa.eq(1).all()
    pred = np.clip(frame[["p1", "p2", "p3"]].to_numpy(float), 1e-6, 1)
    x = np.stack([tasks[k][0] for k in kept])
    y = np.eye(3)[[tasks[k][1] for k in kept]]
    return x, y, pred, np.array([k[0] for k in kept]), len(keys) - len(kept)


def person_average(pid, values):
    return pd.Series(values, index=pid).groupby(level=0).mean().to_numpy()


def correction_power(pidq, pq, pidv, pv, yv):
    fq = person_average(pidq, pq[:, 2])
    fv = person_average(pidv, pv[:, 2])
    y = person_average(pidv, yv[:, 2])
    cov = np.cov(y, fv, ddof=1)
    power = float(np.clip(cov[0, 1] / ((1 + len(y) / len(fq)) * cov[1, 1]), 0, 1)) if cov[1, 1] > 0 else 0.0
    return power


def objective(xq, pq, xv, yv, pv, power):
    x = np.concatenate([xv, xq])
    weights = np.r_[np.full(len(xv), (1-power)/len(xv)), np.full(len(xq), power/len(xq))]
    b = np.einsum("njp,nj->p", xv, yv-power*pv)/len(xv) + power*np.einsum("njp,nj->p", xq, pq)/len(xq)

    def value(beta):
        utility = x @ beta
        prob = softmax(utility, axis=1)
        return float(weights @ logsumexp(utility, axis=1) - b @ beta), np.einsum("n,nj,njp->p", weights, prob, x)-b

    return x, weights, b, value


def support(x, weights, b):
    x, weights = x[weights > 0], weights[weights > 0]
    n, j, p = x.shape
    rank = int(np.linalg.matrix_rank((x[:, :-1]-x[:, -1:]).reshape(-1, p)))
    if rank != p:
        return dict(status="rank_deficient", rank=rank)
    weighted = (weights[:, None, None]*x).reshape(n*j, p)
    eq = np.r_[np.c_[np.kron(np.eye(n), np.ones((1,j))), np.zeros(n)], np.c_[weighted.T, np.zeros(p)]]
    rhs = np.r_[np.ones(n), b]
    fit = linprog(np.r_[np.zeros(n*j), -1.], A_ub=np.c_[-np.eye(n*j), np.ones(n*j)], b_ub=np.zeros(n*j), A_eq=eq, b_eq=rhs, bounds=[(0,1)]*(n*j)+[(0,1/j)], method="highs")
    if fit.success:
        resid = float(np.max(abs(eq @ fit.x-rhs)))
        margin = float(fit.x[-1])
        return dict(status="interior" if margin > 1e-8 else "boundary_or_near_boundary", rank=rank, margin=margin, primal_residual=resid)
    if fit.status == 2:
        select = np.kron(np.eye(n), np.ones((j,1)))
        flat = x.reshape(n*j,p)
        dual = linprog(np.r_[-b,b,weights], A_ub=np.r_[np.c_[flat,-flat,-select],np.r_[np.ones(2*p),np.zeros(n)][None,:]], b_ub=np.r_[np.zeros(n*j),1.], bounds=[(0,None)]*(2*p)+[(None,None)]*n, method="highs")
        if dual.success:
            direction = dual.x[:p]-dual.x[p:2*p]
            gap = float(b@direction - weights@np.max(x@direction,axis=1))
            if gap > 1e-8:
                return dict(status="outside",rank=rank,recession_gap=gap,direction=direction.tolist())
    return dict(status="unresolved",rank=rank,lp_status=int(fit.status))


def main(inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        assert digest((inputs/name).read_bytes()) == expected
    records, coefficients, alignment = [], [], []
    with zipfile.ZipFile(inputs/CORE) as core, zipfile.ZipFile(inputs/OUTPUT) as output:
        read = lambda name: output.read(next(n for n in output.namelist() if n.endswith("/"+name)))
        frozen = json.loads(read("congelamento.json"))
        assert digest(read("previsoes_llm.csv")) == frozen["hash_previsoes"]
        prev = pd.read_csv(io.BytesIO(read("previsoes_llm.csv")))
        published = pd.read_csv(io.BytesIO(read("estimandos_vereditos.csv"))).rename(columns={"lambda":"power_source"})
        assert len(published) == 48 and len(prev) == 7200
        assert len(published[published.metodo.eq("LLM")]) == 24
        for context in sorted(published.contexto.unique()):
            people, tasks, names, idx = read_table(core, context)
            for row in published[published.contexto.eq(context) & published.metodo.eq("LLM")].itertuples(index=False):
                h, v = split_target(people, context, row.rep)
                predictions = prev[prev.contexto.eq(context) & prev.rep.eq(row.rep) & prev.cond.eq(row.cond)].set_index(["pid", "card"])
                xq, yq, pq, pidq, missingq = subset(tasks, predictions, h)
                xv, yv, pv, pidv, missingv = subset(tasks, predictions, v)
                assert len(set(pidq)) == 35 and len(set(pidv)) == 15 and not set(pidq)&set(pidv)
                lam = correction_power(pidq, pq, pidv, pv, yv)
                assert abs(lam-row.power_source) < 1e-12
                original = objective(xq,pq,xv,yv,pv,lam)
                # Same published target optimizer convention, not predictor fitting.
                replay = minimize(original[3], np.zeros(len(names)), jac=True, method="L-BFGS-B")
                error = float(abs(replay.x[idx]-row.cor_beta))
                assert error < 1e-3, (context,row.rep,row.cond,error)
                alignment.append(dict(context=context,rep=int(row.rep),coverage=row.cond,power_error=abs(lam-row.power_source),published_beta=float(row.cor_beta),replayed_beta=float(replay.x[idx]),coefficient_error=error))
                for label,power in [("released_power",lam),("unit_power_sensitivity",1.0)]:
                    x,w,b,fun = objective(xq,pq,xv,yv,pv,power)
                    result = support(x,w,b)
                    record = dict(context=context,rep=int(row.rep),coverage=row.cond,predictor="released_LLM",rule=label,power=power,parameters=len(names),labelled_people=len(set(pidv)),auxiliary_people=len(set(pidq)),labelled_menus=len(xv),auxiliary_menus=len(xq),excluded_menus=missingq+missingv,probability_mass_error=float(max(abs(pq.sum(axis=1)-1).max(),abs(pv.sum(axis=1)-1).max())),**result)
                    if result["status"] == "interior":
                        fit = minimize(fun,np.zeros(len(names)),jac=True,method="BFGS",options={"gtol":1e-10,"maxiter":2000})
                        grad = float(np.max(abs(fun(fit.x)[1])))
                        assert grad < 1e-7
                        # Independent interior witness: every finite softmax is
                        # strictly positive, and its moment matches the target.
                        prob = softmax(x@fit.x,axis=1)
                        record.update(score_residual=grad,softmax_min=float(prob[w>0].min()),optimizer_success=bool(fit.success))
                        coefficients.append(dict(context=context,rep=int(row.rep),coverage=row.cond,rule=label,names=names,beta=fit.x.tolist()))
                    records.append(record)
                    print(json.dumps({k:record[k] for k in ["context","rep","coverage","rule","status"]}),flush=True)
    frame = pd.DataFrame(records)
    frame.drop(columns=["direction"],errors="ignore").to_csv(out/"finite_support.csv",index=False)
    save(out/"support_details.json",records)
    save(out/"coefficients.json",coefficients)
    save(out/"published_alignment.json",alignment)
    save(out/"summary.json",dict(completed=True,published_cells=48,reconstructed_released_LLM_cells=24,unreconstructed_EST_cells=24,unreconstructed_reason="Frozen EST prediction vectors and fitted predictor coefficients are absent from the output archive. Re-fitting that predictor is outside this fixed-output audit.",support_counts={rule:g.status.value_counts().to_dict() for rule,g in frame.groupby("rule")},minimum_margin=float(frame.margin.min()) if "margin" in frame else None,maximum_alignment_coefficient_error=max(a["coefficient_error"] for a in alignment),maximum_alignment_power_error=max(a["power_error"] for a in alignment),source_code_executed=False,new_predictors_trained=0,new_LLM_calls=0,new_responses=0,new_MC_runs=0,source_frozen_prediction_sha256=frozen["hash_previsoes"],python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,script_sha256=digest(Path(__file__).read_bytes())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.inputs,args.out)
