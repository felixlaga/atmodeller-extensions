# SPDX-License-Identifier: Apache-2.0
"""Check analysis provenance, conserved inventories and delivered artifact hashes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
MANIFEST = HERE / "artifact_manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    extensions = {
        ".py",
        ".md",
        ".csv",
        ".json",
        ".npy",
        ".xlsx",
        ".png",
        ".svg",
        ".txt",
        ".pdf",
        ".html",
        ".tex",
        ".bib",
        ".sty",
    }
    return {
        str(p.relative_to(HERE)): digest(p)
        for p in sorted(HERE.rglob("*"))
        if p.is_file()
        and p.suffix in extensions
        and p != MANIFEST
        and "__pycache__" not in p.parts
        and not (p.parent.name == "paper" and p.suffix in {".aux", ".log", ".bbl", ".blg", ".out"})
    }


def recalibration_checks():
    """Post-audit analyses: digitization, independent solver, recalibration, design, draft."""
    from research.spinodal_onset.calibration import Problem
    from research.spinodal_onset.mixing_models import published_gilmore26, weight_to_z

    key = json.loads((OUT / "key_results.json").read_text())["at_2GPa_3wt"]
    m = published_gilmore26()
    z = weight_to_z(0.03)
    assert abs(m.spinodal_temperature(z, 2.0) - key["spinodal_K"]) < 1e-6
    assert abs(m.binodal_temperature(z, 2.0) - key["binodal_K"]) < 1e-4
    assert (
        abs(
            m.rich_hydrogen_fraction(z, key["spinodal_K"], 2.0)
            - key["H_rich_fraction_at_spinodal"]
        )
        < 1e-9
    )
    digitized = json.loads((OUT / "source_figure_digitized.json").read_text())
    assert digitized["wide_gap_marker_max_abs_difference_from_table"] < 0.002
    assert max(c["rms_residual_K"] for c in digitized["curves"].values()) < 5
    calibration = json.loads((OUT / "calibration_summary.json").read_text())
    runs = {(r["model"], r["version"]) for r in calibration["runs"]}
    assert len(runs) == 9, runs
    for run in calibration["runs"]:
        assert run["quantiles"]["interval_K"]["n"] >= 2000
        rhat = [v["rhat"] for v in run["diagnostics"].values() if np.isfinite(v["rhat"])]
        assert max(rhat) < 1.1, (run["model"], run["version"], max(rhat))
    published = Problem("M1", "table").chi2_parts(np.array([75, 424, 1 / 4400, -1 / 35]))
    for k, v in published.items():
        assert abs(v - calibration["published_fit"]["chi2"][k]) < 1e-6, k
    samples = pd.read_csv(OUT / "calibration_samples.csv")
    assert len(samples) == 9 * 2400
    for name in ("design_homogeneous.csv", "design_curvature.csv", "design_summary.json"):
        assert (OUT / name).is_file(), name
    numbers = (HERE / "paper/numbers.tex").read_text()
    assert "\\newcommand{\\pubIntTwo}{414}" in numbers
    assert (HERE / "paper/main.pdf").is_file()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Explicitly record a new successfully validated artifact snapshot",
    )
    args = parser.parse_args()
    provenance = json.loads((OUT / "provenance.json").read_text())
    for name, expected in provenance["sources"].items():
        assert digest(ROOT / name) == expected, f"Analysis source changed: {name}"
    for manifest in ("sources/manifest.json", "sources/figure_manifest.json"):
        for source in json.loads((HERE / manifest).read_text()):
            assert digest(HERE / "sources" / source["file"]) == source["sha256"]
    identity = json.loads((OUT / "identifiability_checks.json").read_text())
    assert digest(HERE / "identifiability.py") == identity["script_sha256"]
    assert identity["homogeneous_EDTable2_states_unchanged"]
    assert identity["maximum_relative_curvature_error_against_autodiff"] < 1e-10
    assert identity["minimum_sampled_support_margin_J_per_fit_mol"] > -1e-7
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["native_solver"]["number_solves"] == 27
    assert summary["native_solver"]["all_accepted"]
    assert summary["native_solver"]["maximum_relative_element_error"] < 1e-10
    assert summary["maximum_lever_rule_error"] < 1e-10
    assert summary["onset_rows"] == len(pd.read_csv(OUT / "onsets.csv")) == 20664
    assert summary["cooling_rows"] == len(pd.read_csv(OUT / "cooling_paths.csv")) == 10827
    assert summary["stress_rows"] == len(pd.read_csv(OUT / "sensitivity.csv")) == 375
    audit = pd.read_csv(OUT / "source_data_audit.csv")
    assert audit.valid.all()
    assert (audit.groupby("model").size() == 14).all()
    assert (audit.groupby("model").observed_two_phase.sum() == 4).all()
    roots = json.loads((OUT / "root_crosschecks.json").read_text())
    assert len(roots) == 36
    assert max(r["binodal_composition_error"] for r in roots) < 1e-7
    assert max(r.get("spinodal_temperature_independent_error_K", 0) for r in roots) < 1e-6
    assert all(not r["above_binodal_two_phase"] and r["below_binodal_two_phase"] for r in roots)
    key = json.loads((OUT / "key_results.json").read_text())
    assert key["maximum_pressure_dependence_excluding_0_1_wt_percent_from_critical"] < 1e-9
    assert key["maximum_pressure_dependence_of_H_fraction_at_spinodal"] < 1e-4
    np.testing.assert_allclose(key["at_2GPa_3wt"]["interval_K"], 413.7381621485756, atol=1e-6)
    assert "288 passed, 3 deselected" in (HERE / "validation/tests.txt").read_text()
    recalibration_checks()
    explorer = (HERE / "explorer.html").read_text()
    embedded = json.loads(explorer.split("const data=", 1)[1].split(";const $", 1)[0])
    paths = pd.read_csv(OUT / "cooling_paths.csv")
    for state, data in embedded.items():
        pressure, weight = (int(v) for v in state.split("_"))
        selected = paths.loc[
            (paths.model == "gilmore26")
            & (paths.pressure_GPa == pressure)
            & (paths.hydrogen_wt_percent == weight)
        ]
        expected_rows = selected[
            ["temperature_K", "H_equilibrium_fraction", "H_delayed_limit_fraction"]
        ].to_numpy()
        np.testing.assert_allclose(data["rows"], expected_rows, atol=5.1e-9, rtol=0)
    actual = snapshot()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({"algorithm": "sha256", "files": actual}, indent=2) + "\n")
    expected = json.loads(MANIFEST.read_text())["files"]
    assert actual.keys() == expected.keys(), (
        f"Artifact set changed: {actual.keys() ^ expected.keys()}"
    )
    differences = [path for path in actual if actual[path] != expected[path]]
    assert not differences, f"Artifact content changed: {differences}"
    print(
        f"PASS: {len(actual)} artifact hashes, {len(provenance['sources'])} analysis-source hashes, original data, 27 native solves, 36 binodal crossings, "
        f"{sum('spinodal_temperature_independent_error_K' in r for r in roots)} independent spinodal roots, conservation, invariance and all 9 interactive curves."
    )


if __name__ == "__main__":
    main()
