from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_published_results.py"


def _load_verifier():
    specification = importlib.util.spec_from_file_location("ursa_result_verifier", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_published_results_recompute_from_case_rows():
    report = _load_verifier().verify_all(REPOSITORY_ROOT)
    assert report["status"] == "pass"
    assert report["three_carrier_45"]["case_count"] == 45
    assert report["bo04_932"]["case_count"] == 932
    route = report["route_325"]["all_325"]
    assert route["case_count"] == 325
    assert route["changed_route_count"] == 19
    assert route["outcome_counts"] == {"better": 13, "same": 306, "worse": 6}
    changed = route["changed_route"]
    assert abs(changed["raw_reference_energy_j"] - 724582.1815317523) < 1.0e-9
    assert abs(changed["selected_reference_energy_j"] - 574706.2393130724) < 1.0e-9
    assert abs(changed["net_saving_j"] - 149875.9422186797) < 1.0e-9
    assert abs(changed["ratio_of_sums_saving_percent"] - 20.684464238665786) < 1.0e-12
