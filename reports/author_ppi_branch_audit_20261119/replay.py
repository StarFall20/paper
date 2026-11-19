"""Compare the authors' pinned research branch on the existing frozen designs."""

import argparse
import hashlib
import importlib
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
BASELINE = next(
    (ROOT.parent / name for name in ("official_ppi_runtime_audit_20261117", "official_ppi_runtime")
     if (ROOT.parent / name / "frozen_input_manifest.json").is_file()),
    ROOT.parent / "official_ppi_runtime_audit_20261117",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream(upstream):
    manifest = read_json(ROOT / "upstream.json")
    expected = set()
    for path, size, blob in manifest["files"]:
        expected.add(path)
        data = (upstream / path).read_bytes()
        assert len(data) == size, path
        header = f"blob {size}\0".encode("ascii")
        assert hashlib.sha1(header + data).hexdigest() == blob, path
    sources = {p.relative_to(upstream).as_posix() for p in (upstream / "ppi_py").rglob("*.py")}
    assert sources == expected, (sources - expected, expected - sources)
    return manifest


def reconstruct_point_step(data, initial, power):
    human = data["X"]
    auxiliary = data["X_unlabeled"]
    human_prob = expit(human @ initial)
    aux_prob = expit(auxiliary @ initial)
    human_curv = human_prob * (1 - human_prob)
    aux_curv = aux_prob * (1 - aux_prob)
    hessian = (
        human.T @ (human_curv[:, None] * human)
        + auxiliary.T @ (aux_curv[:, None] * auxiliary)
    ) / (len(human) + len(auxiliary))
    score = (
        human.T @ (human_prob - data["Y"] - power * (human_prob - data["Yhat"])) / len(human)
        + power * auxiliary.T @ (aux_prob - data["Yhat_unlabeled"]) / len(auxiliary)
    )
    return initial - np.linalg.solve(hessian, score)


def replay(upstream, inputs, out):
    manifest = verify_upstream(upstream)
    frozen = read_json(BASELINE / "frozen_input_manifest.json")
    assert sha256(inputs / "cases.json") == frozen["case_list_sha256"]
    assert sha256(BASELINE / "plan.json") == frozen["plan_sha256"]
    for item in frozen["cases"]:
        assert sha256(inputs / (item["name"] + ".npz")) == item["sha256"]
    options = read_json(BASELINE / "plan.json")["optimizer_options"]
    sys.path.insert(0, str(BASELINE / "analysis"))
    from run_official_api import loss_terms, support, write

    sys.path.insert(0, str(upstream.resolve()))
    package = importlib.import_module("ppi_py.ppi")
    assert Path(package.__file__).resolve() == (upstream / "ppi_py/ppi.py").resolve()
    minimize = package.minimize
    calc_power = package._calc_lam_glm
    opt_trace = []
    power_trace = []

    def trace_minimize(*args, **kwargs):
        result = minimize(*args, **kwargs)
        opt_trace.append({
            "success": bool(result.success),
            "status": int(result.status),
            "iterations": int(result.nit),
            "x": result.x.copy(),
        })
        return result

    def trace_power(*args, **kwargs):
        value = calc_power(*args, **kwargs)
        power_trace.append({"value": float(value), "clip": kwargs.get("clip", False)})
        return value

    def call(kind, power, one_step, data):
        opt_trace.clear()
        power_trace.clear()
        row = {"api": kind, "requested_power": power, "one_step": one_step}
        function = package.ppi_logistic_pointestimate if kind == "point" else package.ppi_logistic_ci
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = np.asarray(function(**data, lam=power, one_step=one_step,
                                             optimizer_options=dict(options)))
                row.update(result=result, returned_finite=bool(np.isfinite(result).all()), exception=None)
            except Exception as exc:
                row.update(result=None, returned_finite=False,
                           exception={"type": type(exc).__name__, "message": str(exc)})
            row["warnings"] = [str(item.message) for item in caught]
        row["optimizer_trace"] = list(opt_trace)
        row["power_trace"] = list(power_trace)
        effective_power = power if power is not None else power_trace[-1]["value"] if power_trace else None
        row["effective_power"] = effective_power
        if kind == "point" and row["returned_finite"]:
            if 0 <= effective_power <= 1:
                _, _, _, _, score = loss_terms(data, effective_power)
                row["score_max"] = float(np.max(np.abs(score(result))))
                row["support"] = support(data, effective_power)["state"]
            if one_step and power is None:
                assert len(opt_trace) == 1 and len(power_trace) == 1
                expected = reconstruct_point_step(data, opt_trace[0]["x"], effective_power)
                row["independent_one_step_error"] = float(np.max(np.abs(expected - result)))
                np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-9)
        if kind == "ci" and row["returned_finite"]:
            row["interval_center_minus_final_optimizer_max"] = float(
                np.max(np.abs(result.mean(axis=0) - opt_trace[-1]["x"]))
            )
        return row

    rows = []
    package.minimize = trace_minimize
    package._calc_lam_glm = trace_power
    try:
        for case in read_json(inputs / "cases.json"):
            with np.load(inputs / case["input_file"], allow_pickle=False) as source:
                data = {key: source[key] for key in source.files}
            requests = [("point", power, False) for power in case["powers"]]
            if case["kind"] == "public":
                requests.extend([("point", None, True), ("ci", 1, False),
                                 ("ci", None, False), ("ci", None, True)])
            else:
                requests.append(("ci", case["powers"][0], False))
            for args in requests:
                row = call(*args, data)
                row.update(case=case["name"], kind=case["kind"])
                rows.append(row)
    finally:
        package.minimize = minimize
        package._calc_lam_glm = calc_power
    assert len(rows) == 136
    verify_upstream(upstream)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "records.json", rows)
    write(out / "environment.json", {
        "upstream_commit": manifest["commit"],
        "source_before_after_verified": True,
        "plan_sha256": frozen["plan_sha256"],
        "frozen_case_list_sha256": frozen["case_list_sha256"],
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "numba", "statsmodels")},
        "python": sys.version,
    })
    print(f"Replayed {len(rows)} calls from {manifest['commit']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(args.upstream, args.inputs, args.out)
