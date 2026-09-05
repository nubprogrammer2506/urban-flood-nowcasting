from pathlib import Path
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPTH_TIMESERIES_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_flood_depth_timeseries.csv"
DEPTH_PEAK_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_flood_depth_peak.geojson"
RISK_TIMESERIES_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_flood_risk_timeseries.csv"
RISK_PEAK_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_flood_risk_peak.geojson"
RISK_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_graph_flood_risk.graphml"

MINIMAL_MAX_M = 0.02
LOW_MAX_M = 0.10
MODERATE_MAX_M = 0.30
POLICY_NAME = "prototype_depth_band_v1"

RISK_RANK = {
    "unknown": -1,
    "minimal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
}

def expected_risk(status, depth):
    if str(status) != "mapped":
        return "unknown"
    value = float(depth)
    if not np.isfinite(value) or value < 0:
        return "unknown"
    if value <= MINIMAL_MAX_M:
        return "minimal"
    if value <= LOW_MAX_M:
        return "low"
    if value <= MODERATE_MAX_M:
        return "moderate"
    return "high"

def main():
    print("\n=== ROAD FLOOD-RISK CLASSIFICATION VALIDATION ===\n")

    for path in [
        DEPTH_TIMESERIES_PATH,
        DEPTH_PEAK_PATH,
        RISK_TIMESERIES_PATH,
        RISK_PEAK_PATH,
        RISK_GRAPH_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    depth_ts = pd.read_csv(DEPTH_TIMESERIES_PATH)
    depth_peak = gpd.read_file(DEPTH_PEAK_PATH)
    risk_ts = pd.read_csv(RISK_TIMESERIES_PATH)
    risk_peak = gpd.read_file(RISK_PEAK_PATH)
    graph = nx.read_graphml(RISK_GRAPH_PATH, force_multigraph=True)

    errors = []

    print(f"Depth timeseries rows   : {len(depth_ts)}")
    print(f"Risk timeseries rows    : {len(risk_ts)}")
    print(f"Depth peak roads        : {len(depth_peak)}")
    print(f"Risk peak roads         : {len(risk_peak)}")
    print(f"Risk graph edges        : {graph.number_of_edges()}")

    if len(risk_ts) != len(depth_ts):
        errors.append("risk timeseries row count changed")
    if len(risk_peak) != len(depth_peak):
        errors.append("risk peak feature count changed")
    if graph.number_of_edges() != len(depth_peak):
        errors.append("risk graph edge count mismatch")

    duplicate_ts = int(risk_ts.duplicated(subset=["edge_id", "interval_id"]).sum())
    duplicate_peak = int(risk_peak.duplicated(subset=["edge_id"]).sum())

    print(f"Duplicate timeseries    : {duplicate_ts}")
    print(f"Duplicate peak IDs      : {duplicate_peak}")

    if duplicate_ts:
        errors.append("duplicate risk timeseries rows")
    if duplicate_peak:
        errors.append("duplicate risk peak IDs")

    wrong_ts_class = wrong_ts_rank = wrong_ts_policy = 0
    for row in risk_ts.itertuples(index=False):
        expected = expected_risk(row.sampling_status, row.max_depth_m)
        if str(row.flood_risk) != expected:
            wrong_ts_class += 1
        if int(row.flood_risk_rank) != RISK_RANK[expected]:
            wrong_ts_rank += 1
        if str(row.risk_policy) != POLICY_NAME:
            wrong_ts_policy += 1

    wrong_peak_class = wrong_peak_rank = wrong_peak_policy = 0
    for row in risk_peak.itertuples(index=False):
        expected = expected_risk(row.sampling_status, row.scenario_peak_max_depth_m)
        if str(row.flood_risk) != expected:
            wrong_peak_class += 1
        if int(row.flood_risk_rank) != RISK_RANK[expected]:
            wrong_peak_rank += 1
        if str(row.risk_policy) != POLICY_NAME:
            wrong_peak_policy += 1

    print("\nClassification checks")
    print(f"Wrong interval classes : {wrong_ts_class}")
    print(f"Wrong interval ranks   : {wrong_ts_rank}")
    print(f"Wrong interval policy  : {wrong_ts_policy}")
    print(f"Wrong peak classes     : {wrong_peak_class}")
    print(f"Wrong peak ranks       : {wrong_peak_rank}")
    print(f"Wrong peak policy      : {wrong_peak_policy}")

    if wrong_ts_class:
        errors.append("interval risk classification mismatch")
    if wrong_ts_rank:
        errors.append("interval risk rank mismatch")
    if wrong_ts_policy:
        errors.append("interval risk policy mismatch")
    if wrong_peak_class:
        errors.append("peak risk classification mismatch")
    if wrong_peak_rank:
        errors.append("peak risk rank mismatch")
    if wrong_peak_policy:
        errors.append("peak risk policy mismatch")

    changed_passability = 0
    changed_cost = 0
    missing_risk_attrs = 0

    for _, _, _, data in graph.edges(keys=True, data=True):
        if int(float(data.get("is_passable", 0))) != 1:
            changed_passability += 1

        length_m = float(data["length_m"])
        routing_cost = float(data["routing_cost"])
        if abs(routing_cost - length_m) > 1e-6:
            changed_cost += 1

        if (
            "flood_risk" not in data
            or "flood_risk_rank" not in data
            or "risk_policy" not in data
        ):
            missing_risk_attrs += 1

    risk_flag = int(float(graph.graph.get("risk_policy_applied", 0)))
    passability_flag = int(float(graph.graph.get("passability_policy_applied", 1)))
    safety_standard_flag = int(float(graph.graph.get("risk_policy_is_safety_standard", 1)))

    print("\nPolicy separation")
    print(f"Changed passability    : {changed_passability}")
    print(f"Changed routing costs  : {changed_cost}")
    print(f"Missing risk attrs     : {missing_risk_attrs}")
    print(f"Risk policy flag       : {risk_flag}")
    print(f"Passability flag       : {passability_flag}")
    print(f"Safety-standard flag   : {safety_standard_flag}")

    if changed_passability:
        errors.append("passability changed during risk milestone")
    if changed_cost:
        errors.append("routing cost changed during risk milestone")
    if missing_risk_attrs:
        errors.append("risk graph missing risk attributes")
    if risk_flag != 1:
        errors.append("risk policy flag not applied")
    if passability_flag != 0:
        errors.append("passability policy applied too early")
    if safety_standard_flag != 0:
        errors.append("prototype risk policy marked as safety standard")

    print("\nScenario peak risk distribution")
    for label in ["minimal", "low", "moderate", "high", "unknown"]:
        print(f"  {label:<8}: {int((risk_peak['flood_risk'] == label).sum())}")

    print("\nIMPORTANT:")
    print("Risk classes are prototype scenario depth bands, not municipal safety standards.")
    print("Passability and flood-aware routing-cost policy must be validated separately.")

    print("\n============================================")
    if errors:
        print("ROAD FLOOD-RISK CLASSIFICATION VALIDATION: FAILED")
        for error in errors:
            print(f" - {error}")
    else:
        print("ROAD FLOOD-RISK CLASSIFICATION VALIDATION: PASSED")
    print("============================================\n")

if __name__ == "__main__":
    main()
