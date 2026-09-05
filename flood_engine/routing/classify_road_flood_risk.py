from pathlib import Path
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPTH_TIMESERIES_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_flood_depth_timeseries.csv"
DEPTH_PEAK_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_flood_depth_peak.geojson"
DEPTH_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "road_graph_flood_depth.graphml"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "roads"
RISK_TIMESERIES_OUTPUT = OUTPUT_DIR / "road_flood_risk_timeseries.csv"
RISK_PEAK_OUTPUT = OUTPUT_DIR / "road_flood_risk_peak.geojson"
RISK_GRAPH_OUTPUT = OUTPUT_DIR / "road_graph_flood_risk.graphml"

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

def classify_depth(depth_m):
    if depth_m is None:
        return "unknown"
    depth = float(depth_m)
    if not np.isfinite(depth) or depth < 0:
        return "unknown"
    if depth <= MINIMAL_MAX_M:
        return "minimal"
    if depth <= LOW_MAX_M:
        return "low"
    if depth <= MODERATE_MAX_M:
        return "moderate"
    return "high"

def main():
    print("\n=== ROAD FLOOD-RISK CLASSIFICATION ===\n")

    for path in [DEPTH_TIMESERIES_PATH, DEPTH_PEAK_PATH, DEPTH_GRAPH_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    depth_ts = pd.read_csv(DEPTH_TIMESERIES_PATH)
    depth_peak = gpd.read_file(DEPTH_PEAK_PATH)
    graph = nx.read_graphml(DEPTH_GRAPH_PATH, force_multigraph=True)

    risk_ts = depth_ts.copy()
    risk_ts["flood_risk"] = [
        classify_depth(depth) if str(status) == "mapped" else "unknown"
        for status, depth in zip(risk_ts["sampling_status"], risk_ts["max_depth_m"])
    ]
    risk_ts["flood_risk_rank"] = risk_ts["flood_risk"].map(RISK_RANK).astype(int)
    risk_ts["risk_policy"] = POLICY_NAME
    risk_ts.to_csv(RISK_TIMESERIES_OUTPUT, index=False)

    risk_peak = depth_peak.copy()
    risk_peak["flood_risk"] = [
        classify_depth(depth) if str(status) == "mapped" else "unknown"
        for status, depth in zip(
            risk_peak["sampling_status"],
            risk_peak["scenario_peak_max_depth_m"],
        )
    ]
    risk_peak["flood_risk_rank"] = risk_peak["flood_risk"].map(RISK_RANK).astype(int)
    risk_peak["risk_policy"] = POLICY_NAME
    risk_peak.to_file(RISK_PEAK_OUTPUT, driver="GeoJSON")

    risk_lookup = {str(row.edge_id): row for row in risk_peak.itertuples(index=False)}

    for _, _, _, data in graph.edges(keys=True, data=True):
        edge_id = str(data.get("road_id", ""))
        if edge_id not in risk_lookup:
            raise ValueError(f"Missing risk record for edge {edge_id}")

        record = risk_lookup[edge_id]
        risk_class = str(record.flood_risk)
        peak_depth = float(record.scenario_peak_max_depth_m)

        data["flood_depth_m"] = peak_depth
        data["flood_risk"] = risk_class
        data["flood_risk_rank"] = int(RISK_RANK[risk_class])
        data["risk_policy"] = POLICY_NAME

        # P2.18 only: do not change passability or routing cost yet.
        data["is_passable"] = int(float(data.get("is_passable", 1)))
        data["routing_cost"] = float(data.get("routing_cost", data["length_m"]))

    graph.graph["risk_policy_applied"] = 1
    graph.graph["risk_policy"] = POLICY_NAME
    graph.graph["risk_policy_is_safety_standard"] = 0
    graph.graph["passability_policy_applied"] = 0

    nx.write_graphml(graph, RISK_GRAPH_OUTPUT)

    mapped_peak = risk_peak[risk_peak["sampling_status"] == "mapped"]

    print(f"Road edges              : {len(risk_peak)}")
    print(f"Mapped road edges       : {len(mapped_peak)}")
    print(f"Unmapped road edges     : {len(risk_peak) - len(mapped_peak)}")
    print(f"Risk policy             : {POLICY_NAME}")

    print("\nScenario peak risk classes")
    for risk_class in ["minimal", "low", "moderate", "high", "unknown"]:
        count = int((risk_peak["flood_risk"] == risk_class).sum())
        print(f"  {risk_class:<8}: {count}")

    print("\nThresholds")
    print("  minimal  : depth <= 0.02 m")
    print("  low      : 0.02 < depth <= 0.10 m")
    print("  moderate : 0.10 < depth <= 0.30 m")
    print("  high     : depth > 0.30 m")

    print("\nSaved:")
    print(RISK_TIMESERIES_OUTPUT)
    print(RISK_PEAK_OUTPUT)
    print(RISK_GRAPH_OUTPUT)

    print("\nIMPORTANT:")
    print("These are prototype scenario risk bands, not municipal road-safety standards.")
    print("Passability and routing-cost policy are intentionally NOT applied in this milestone.")
    print("\nRoad flood-risk classification completed.")

if __name__ == "__main__":
    main()
