from collections import defaultdict, deque
from pathlib import Path
import re

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNOFF_DIR = PROJECT_ROOT / "data" / "processed" / "rainfall"
FLOW_DIRECTION_PATH = PROJECT_ROOT / "data" / "processed" / "terrain" / "rajendra_nagar_flow_direction.tif"
INLETS_PATH = PROJECT_ROOT / "data" / "processed" / "drainage" / "inferred_inlets.geojson"
BASE_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "drainage" / "drainage_graph.graphml"
CAPACITY_GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "drainage" / "drainage_capacity.graphml"
CAPACITY_GEOJSON_PATH = PROJECT_ROOT / "data" / "processed" / "drainage" / "drainage_capacity.geojson"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "coupling"
INLET_OUTPUT = OUTPUT_DIR / "inlet_interval_coupling.csv"
DRAIN_OUTPUT = OUTPUT_DIR / "drainage_interval_routing.csv"
MASS_OUTPUT = OUTPUT_DIR / "interval_mass_balance.csv"
OVERFLOW_OUTPUT = OUTPUT_DIR / "drainage_overflow_points.geojson"
MAPPING_AUDIT_OUTPUT = OUTPUT_DIR / "inlet_surface_mapping.csv"
EXPECTED_CRS = CRS.from_epsg(32643)

DIRECTION_OFFSETS = {
    1: (-1, 1), 2: (-1, 0), 3: (-1, -1), 4: (0, -1),
    5: (1, -1), 6: (1, 0), 7: (1, 1), 8: (0, 1),
}

CAPACITY_FIELDS = [
    "strahler_order", "capacity_class", "scenario_width_m",
    "scenario_depth_m", "scenario_manning_n",
    "capacity_effective_slope", "capacity_slope_floor",
    "capacity_m3s", "capacity_scenario", "capacity_source",
    "capacity_is_observed",
]


def crs_equals(a, b):
    if a is None or b is None:
        return False
    return CRS.from_user_input(a).equals(CRS.from_user_input(b))


def parse_interval(path, tags):
    if tags.get("start_minutes") is not None and tags.get("end_minutes") is not None:
        start_min = int(float(tags["start_minutes"]))
        end_min = int(float(tags["end_minutes"]))
    else:
        match = re.search(r"runoff_volume_(\d+)_(\d+)\.tif$", path.name)
        if match is None:
            raise ValueError(f"Cannot determine interval from {path.name}")
        start_min, end_min = map(int, match.groups())
    if end_min <= start_min:
        raise ValueError(f"Invalid interval in {path.name}: {start_min}->{end_min}")
    return {
        "interval_id": f"{start_min:03d}_{end_min:03d}",
        "scenario_id": str(tags.get("scenario") or tags.get("scenario_id") or "unknown"),
        "start_min": start_min,
        "end_min": end_min,
        "duration_s": (end_min - start_min) * 60,
    }


def load_capacity_graph():
    if CAPACITY_GRAPH_PATH.exists():
        graph = nx.read_graphml(CAPACITY_GRAPH_PATH, force_multigraph=True)
    else:
        if not BASE_GRAPH_PATH.exists():
            raise FileNotFoundError("Run build_drainage_graph.py first; drainage_graph.graphml is missing.")
        if not CAPACITY_GEOJSON_PATH.exists():
            raise FileNotFoundError("Run build_drainage_capacity.py first; drainage_capacity.geojson is missing.")
        graph = nx.read_graphml(BASE_GRAPH_PATH, force_multigraph=True)
        capacity = gpd.read_file(CAPACITY_GEOJSON_PATH)
        missing = [f for f in ["drain_id"] + CAPACITY_FIELDS if f not in capacity.columns]
        if missing:
            raise ValueError("Capacity GeoJSON missing fields: " + ", ".join(missing))
        lookup = capacity.set_index("drain_id")[CAPACITY_FIELDS].to_dict("index")
        for _, _, _, data in graph.edges(keys=True, data=True):
            drain_id = str(data.get("source_drain_id", data.get("drain_id", "")))
            if drain_id not in lookup:
                raise ValueError(f"No capacity record for {drain_id}")
            for field, value in lookup[drain_id].items():
                if not pd.isna(value):
                    data[field] = value

    if not graph.is_directed() or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Drainage graph must be a directed acyclic graph.")
    branching = [node for node in graph.nodes if graph.out_degree(node) > 1]
    if branching:
        raise ValueError(f"Unsupported drainage branching at {len(branching)} nodes.")
    for _, _, _, data in graph.edges(keys=True, data=True):
        if "capacity_m3s" not in data:
            raise ValueError("Drainage edge missing capacity_m3s.")
        value = float(data["capacity_m3s"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Drainage capacity must be finite and positive.")
    return graph


def build_surface_topology(direction, valid_cells):
    height, width = direction.shape
    downstream = {}
    indegree = np.zeros((height, width), dtype=np.int32)
    outlets = []
    rows, cols = np.where(valid_cells)

    for row, col in zip(rows, cols):
        code = int(direction[row, col])
        if code not in DIRECTION_OFFSETS:
            raise ValueError(f"Invalid direction {code} at ({row},{col})")
        dr, dc = DIRECTION_OFFSETS[code]
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= height or nc < 0 or nc >= width or not valid_cells[nr, nc]:
            downstream[(row, col)] = None
            outlets.append((row, col))
        else:
            downstream[(row, col)] = (nr, nc)
            indegree[nr, nc] += 1

    queue = deque((r, c) for r, c in zip(rows, cols) if indegree[r, c] == 0)
    order = []
    work = indegree.copy()
    while queue:
        cell = queue.popleft()
        order.append(cell)
        target = downstream[cell]
        if target is None:
            continue
        nr, nc = target
        work[nr, nc] -= 1
        if work[nr, nc] == 0:
            queue.append((nr, nc))

    if len(order) != int(valid_cells.sum()):
        raise RuntimeError("Surface flow topology contains a cycle or unresolved cells.")
    return downstream, order, outlets


def map_inlets_to_cells(inlets, raster, valid_cells):
    """
    Map road-proximity eligible inferred inlet candidates to the
    canonical 30 m runoff grid.

    Important:
    drainage lines were vector-clipped to the AOI, so some graph
    endpoints can lie exactly on the polygon boundary. With the
    runoff mask using cell-centre inclusion, such a boundary point
    can index to a raster cell that is outside the valid runoff AOI.

    We do NOT snap those points to another cell, because that could
    capture runoff from the wrong D8 path. They remain valid inferred
    drainage/inlet candidates, but are excluded from this raster
    coupling interface and recorded in an audit table.

    If multiple candidates fall in the same 30 m cell, only one can
    be represented by the surface grid without double counting. The
    candidate closest to the cell centre is selected deterministically;
    the others are recorded as excluded_same_30m_cell.
    """

    if not crs_equals(inlets.crs, raster.crs):
        inlets = inlets.to_crs(raster.crs)

    eligible = inlets[
        inlets["coupling_eligible"].astype(int) == 1
    ].copy()

    if eligible.empty:
        raise ValueError("No coupling-eligible inlet candidates.")

    candidate_records = []
    audit_records = []

    for idx, row in eligible.iterrows():
        geom = row.geometry

        base_audit = {
            "inlet_id": str(row["inlet_id"]),
            "drain_node_id": str(row["drain_node_id"]),
            "road_coupling_eligible": 1,
            "surface_mapping_selected": 0,
            "surface_mapping_status": "",
            "surface_row": -1,
            "surface_col": -1,
        }

        if geom is None or geom.is_empty:
            base_audit["surface_mapping_status"] = "excluded_invalid_geometry"
            audit_records.append(base_audit)
            continue

        rr, cc = raster.index(geom.x, geom.y)

        if (
            rr < 0
            or rr >= raster.height
            or cc < 0
            or cc >= raster.width
        ):
            base_audit["surface_mapping_status"] = "excluded_outside_grid"
            audit_records.append(base_audit)
            continue

        base_audit["surface_row"] = int(rr)
        base_audit["surface_col"] = int(cc)

        if not valid_cells[rr, cc]:
            base_audit["surface_mapping_status"] = "excluded_outside_runoff_aoi"
            audit_records.append(base_audit)
            continue

        # Distance to the centre of the raster cell is used only as a
        # deterministic tie-breaker if multiple inlet candidates map
        # to the same 30 m surface cell.
        x_center, y_center = raster.xy(rr, cc, offset="center")
        centre_distance = float(
            ((geom.x - x_center) ** 2 + (geom.y - y_center) ** 2) ** 0.5
        )

        candidate_records.append(
            {
                "source_index": idx,
                "surface_row": int(rr),
                "surface_col": int(cc),
                "centre_distance_m": centre_distance,
            }
        )

        base_audit["surface_mapping_status"] = "candidate_valid_cell"
        audit_records.append(base_audit)

    audit = pd.DataFrame(audit_records)

    if not candidate_records:
        raise ValueError(
            "None of the coupling-eligible inlet candidates map to "
            "valid runoff cells."
        )

    candidates = pd.DataFrame(candidate_records)

    # One surface grid cell must feed at most one drainage node.
    # Select the closest candidate to the cell centre, then inlet_id
    # as a deterministic secondary key.
    candidates["inlet_id"] = [
        str(eligible.loc[idx, "inlet_id"])
        for idx in candidates["source_index"]
    ]

    candidates = candidates.sort_values(
        by=[
            "surface_row",
            "surface_col",
            "centre_distance_m",
            "inlet_id",
        ]
    )

    selected = candidates.drop_duplicates(
        subset=["surface_row", "surface_col"],
        keep="first",
    ).copy()

    selected_indices = set(selected["source_index"].tolist())

    for i, audit_row in audit.iterrows():
        inlet_id = str(audit_row["inlet_id"])

        matching_index = eligible.index[
            eligible["inlet_id"].astype(str) == inlet_id
        ]

        if len(matching_index) != 1:
            raise ValueError(
                f"Could not resolve unique source inlet {inlet_id}."
            )

        source_index = matching_index[0]

        if audit.at[i, "surface_mapping_status"] != "candidate_valid_cell":
            continue

        if source_index in selected_indices:
            audit.at[i, "surface_mapping_selected"] = 1
            audit.at[i, "surface_mapping_status"] = "mapped"
        else:
            audit.at[i, "surface_mapping_status"] = "excluded_same_30m_cell"

    mapped = eligible.loc[
        list(selected_indices)
    ].copy()

    selected_lookup = selected.set_index("source_index")

    mapped["surface_row"] = [
        int(selected_lookup.loc[idx, "surface_row"])
        for idx in mapped.index
    ]

    mapped["surface_col"] = [
        int(selected_lookup.loc[idx, "surface_col"])
        for idx in mapped.index
    ]

    mapped = mapped.sort_values(
        by=["drain_node_id", "inlet_id"]
    ).reset_index(drop=True)

    audit = audit.sort_values(
        by=["inlet_id"]
    ).reset_index(drop=True)

    return mapped, audit


def one_out_edge(graph, node):
    edges = list(graph.out_edges(node, keys=True, data=True))
    if not edges:
        return None
    if len(edges) != 1:
        raise ValueError(f"Drain node {node} has {len(edges)} outgoing edges.")
    return edges[0]


def route_surface_to_first_inlet(local_runoff, valid_cells, downstream, order, inlet_cells):
    accumulated = np.zeros(local_runoff.shape, dtype=np.float64)
    accumulated[valid_cells] = local_runoff[valid_cells].astype(np.float64)
    generated = float(accumulated[valid_cells].sum())
    captured = {cell: 0.0 for cell in inlet_cells}
    exported = 0.0

    for cell in order:
        row, col = cell
        volume = float(accumulated[row, col])
        if cell in inlet_cells:
            captured[cell] += volume
            continue
        target = downstream[cell]
        if target is None:
            exported += volume
        else:
            nr, nc = target
            accumulated[nr, nc] += volume

    captured_total = float(sum(captured.values()))
    error = abs(generated - captured_total - exported)
    tolerance = max(0.05, generated * 1e-8)
    if error > tolerance:
        raise RuntimeError(f"Surface mass balance failed: {error:.6f} m3")
    return generated, captured, captured_total, exported, error


def route_drainage(graph, local_inlet_q, interval):
    upstream_q = defaultdict(float)
    rows = []
    overflow_features = []
    total_input_q = float(sum(local_inlet_q.values()))
    total_overflow_q = 0.0
    total_export_q = 0.0

    for node in nx.topological_sort(graph):
        local_q = float(local_inlet_q.get(node, 0.0))
        incoming_q = float(upstream_q.get(node, 0.0))
        total_q = local_q + incoming_q
        outgoing = one_out_edge(graph, node)

        if outgoing is None:
            drain_id = ""
            capacity = np.nan
            source = ""
            observed = np.nan
            conveyed = 0.0
            overflow = 0.0
            export = total_q
            total_export_q += export
        else:
            _, downstream_node, _, data = outgoing
            drain_id = str(data.get("source_drain_id", data.get("drain_id", "")))
            capacity = float(data["capacity_m3s"])
            source = str(data.get("capacity_source", ""))
            observed = int(float(data.get("capacity_is_observed", 0)))
            conveyed = min(total_q, capacity)
            overflow = max(0.0, total_q - conveyed)
            export = 0.0
            upstream_q[downstream_node] += conveyed
            total_overflow_q += overflow

        overflow_volume = overflow * interval["duration_s"]
        rows.append({
            **interval,
            "drain_node_id": str(node),
            "local_inlet_q_m3s": local_q,
            "upstream_q_m3s": incoming_q,
            "total_q_m3s": total_q,
            "outgoing_drain_id": drain_id,
            "capacity_m3s": capacity,
            "capacity_source": source,
            "capacity_is_observed": observed,
            "conveyed_q_m3s": conveyed,
            "overflow_q_m3s": overflow,
            "export_q_m3s": export,
            "overflow_volume_m3": overflow_volume,
        })

        if overflow > 0:
            nd = graph.nodes[node]
            overflow_features.append({
                **interval,
                "drain_node_id": str(node),
                "outgoing_drain_id": drain_id,
                "total_q_m3s": total_q,
                "capacity_m3s": capacity,
                "overflow_q_m3s": overflow,
                "overflow_volume_m3": overflow_volume,
                "capacity_source": source,
                "capacity_is_observed": observed,
                "source_type": "prototype_drainage_capacity_overflow",
                "geometry": Point(float(nd["x"]), float(nd["y"])),
            })

    duration = interval["duration_s"]
    input_vol = total_input_q * duration
    overflow_vol = total_overflow_q * duration
    export_vol = total_export_q * duration
    error = abs(input_vol - overflow_vol - export_vol)
    tolerance = max(0.05, input_vol * 1e-8)
    if error > tolerance:
        raise RuntimeError(f"Drainage mass balance failed: {error:.6f} m3")
    return rows, overflow_features, input_vol, overflow_vol, export_vol, error


def main():
    print("\n=== RUNOFF-DRAINAGE COUPLING ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runoff_files = sorted(RUNOFF_DIR.glob("runoff_volume_*.tif"))
    if not runoff_files:
        raise FileNotFoundError(f"No runoff rasters found in {RUNOFF_DIR}")
    if not FLOW_DIRECTION_PATH.exists() or not INLETS_PATH.exists():
        raise FileNotFoundError("Flow direction or inferred inlet dataset is missing.")

    graph = load_capacity_graph()
    inlets = gpd.read_file(INLETS_PATH)
    required = ["inlet_id", "drain_node_id", "drain_id", "road_id", "coupling_eligible"]
    missing = [f for f in required if f not in inlets.columns]
    if missing:
        raise ValueError("Inlet dataset missing fields: " + ", ".join(missing))

    with rasterio.open(runoff_files[0]) as template:
        if not crs_equals(template.crs, EXPECTED_CRS):
            raise ValueError("Runoff raster is not EPSG:32643.")
        first = template.read(1)
        nodata = template.nodata
        valid_cells = np.isfinite(first) if nodata is None else (np.isfinite(first) & (first != nodata))
        t_crs, t_transform = template.crs, template.transform
        t_width, t_height = template.width, template.height
        eligible, mapping_audit = map_inlets_to_cells(
            inlets, template, valid_cells
        )

    if any(str(node) not in graph for node in eligible["drain_node_id"].astype(str)):
        raise ValueError("One or more eligible inlets reference missing drainage graph nodes.")

    with rasterio.open(FLOW_DIRECTION_PATH) as src:
        if (not crs_equals(src.crs, t_crs) or src.width != t_width or src.height != t_height
                or not src.transform.almost_equals(t_transform)):
            raise ValueError("Flow-direction raster is not aligned with runoff rasters.")
        direction = src.read(1)

    downstream, order, outlets = build_surface_topology(direction, valid_cells)
    inlet_cells = {(int(r.surface_row), int(r.surface_col)) for r in eligible.itertuples(index=False)}

    original_eligible_count = int(
        (inlets["coupling_eligible"].astype(int) == 1).sum()
    )

    outside_aoi_count = int(
        (mapping_audit["surface_mapping_status"] == "excluded_outside_runoff_aoi").sum()
    )

    outside_grid_count = int(
        (mapping_audit["surface_mapping_status"] == "excluded_outside_grid").sum()
    )

    duplicate_cell_count = int(
        (mapping_audit["surface_mapping_status"] == "excluded_same_30m_cell").sum()
    )

    print(f"Runoff intervals       : {len(runoff_files)}")
    print(f"Surface AOI cells      : {int(valid_cells.sum())}")
    print(f"Surface outlet cells   : {len(outlets)}")
    print(f"Road-eligible inlets   : {original_eligible_count}")
    print(f"Raster-mapped inlets   : {len(eligible)}")
    print(f"Excluded outside AOI   : {outside_aoi_count}")
    print(f"Excluded outside grid  : {outside_grid_count}")
    print(f"Duplicate-cell excluded: {duplicate_cell_count}")
    print(f"Drainage nodes         : {graph.number_of_nodes()}")
    print(f"Drainage edges         : {graph.number_of_edges()}\n")

    inlet_rows, drain_rows, overflow_rows, mass_rows = [], [], [], []

    for runoff_path in runoff_files:
        with rasterio.open(runoff_path) as src:
            if (not crs_equals(src.crs, t_crs) or src.width != t_width or src.height != t_height
                    or not src.transform.almost_equals(t_transform)):
                raise ValueError(f"{runoff_path.name} is not aligned with coupling grid.")
            local_runoff = src.read(1)
            interval = parse_interval(runoff_path, src.tags())
            nodata = src.nodata
            current_valid = np.isfinite(local_runoff) if nodata is None else (np.isfinite(local_runoff) & (local_runoff != nodata))
            if not np.array_equal(current_valid, valid_cells):
                raise ValueError(f"{runoff_path.name} has a different valid-cell mask.")

        generated, captured, captured_total, surface_export, surface_error = route_surface_to_first_inlet(
            local_runoff, valid_cells, downstream, order, inlet_cells
        )

        local_inlet_q = defaultdict(float)
        duration = interval["duration_s"]
        for inlet in eligible.itertuples(index=False):
            cell = (int(inlet.surface_row), int(inlet.surface_col))
            arrival = float(captured[cell])
            q = arrival / duration
            node = str(inlet.drain_node_id)
            local_inlet_q[node] += q
            outgoing = one_out_edge(graph, node)
            if outgoing is None:
                raise ValueError(f"Eligible inlet {inlet.inlet_id} maps to a sink node.")
            edge_data = outgoing[3]
            inlet_rows.append({
                **interval,
                "inlet_id": str(inlet.inlet_id),
                "drain_node_id": node,
                "drain_id": str(inlet.drain_id),
                "road_id": str(inlet.road_id),
                "surface_row": int(inlet.surface_row),
                "surface_col": int(inlet.surface_col),
                "arrival_volume_m3": arrival,
                "potential_inflow_q_m3s": q,
                "entry_capacity_m3s": float(edge_data["capacity_m3s"]),
                "entry_capacity_source": str(edge_data.get("capacity_source", "")),
                "entry_capacity_is_observed": int(float(edge_data.get("capacity_is_observed", 0))),
                "coupling_method": "first_eligible_inlet_on_d8_path",
            })

        dr_rows, ov_rows, drain_input, drain_overflow, drain_export, drain_error = route_drainage(
            graph, local_inlet_q, interval
        )
        drain_rows.extend(dr_rows)
        overflow_rows.extend(ov_rows)
        mass_rows.append({
            **interval,
            "generated_surface_volume_m3": generated,
            "captured_to_drain_volume_m3": captured_total,
            "surface_export_volume_m3": surface_export,
            "surface_balance_error_m3": surface_error,
            "drainage_input_volume_m3": drain_input,
            "drainage_export_volume_m3": drain_export,
            "drainage_overflow_volume_m3": drain_overflow,
            "drainage_balance_error_m3": drain_error,
            "surface_capture_fraction": captured_total / generated if generated > 0 else 0.0,
        })

        print(
            f"{interval['start_min']:>3}-{interval['end_min']:>3} min | "
            f"generated {generated:>10.2f} m3 | captured {captured_total:>10.2f} m3 | "
            f"surface export {surface_export:>10.2f} m3 | drain overflow {drain_overflow:>10.2f} m3 | "
            f"drain export {drain_export:>10.2f} m3"
        )

    pd.DataFrame(inlet_rows).to_csv(INLET_OUTPUT, index=False)
    pd.DataFrame(drain_rows).to_csv(DRAIN_OUTPUT, index=False)
    pd.DataFrame(mass_rows).to_csv(MASS_OUTPUT, index=False)
    mapping_audit.to_csv(MAPPING_AUDIT_OUTPUT, index=False)

    if overflow_rows:
        gpd.GeoDataFrame(overflow_rows, geometry="geometry", crs="EPSG:32643").to_file(
            OVERFLOW_OUTPUT, driver="GeoJSON"
        )
    else:
        OVERFLOW_OUTPUT.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    print("\nSaved:")
    print(INLET_OUTPUT)
    print(DRAIN_OUTPUT)
    print(MASS_OUTPUT)
    print(MAPPING_AUDIT_OUTPUT)
    print(OVERFLOW_OUTPUT)
    print("\nIMPORTANT:")
    print("Surface runoff is captured only at the FIRST raster-mapped eligible inferred inlet on each D8 flow path.")
    print("Boundary candidates that are not addressable by the runoff cell-centre mask are audited, not snapped.")
    print("Drainage routing is an instantaneous, no-storage prototype scenario.")
    print("Drain capacities are synthetic Manning-scenario capacities, not surveyed municipal capacities.")
    print("\nRunoff-drainage coupling completed.")


if __name__ == "__main__":
    main()
