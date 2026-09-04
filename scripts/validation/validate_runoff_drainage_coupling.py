from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INLETS_PATH = PROJECT_ROOT / "data" / "processed" / "drainage" / "inferred_inlets.geojson"
INLET_PATH = PROJECT_ROOT / "data" / "processed" / "coupling" / "inlet_interval_coupling.csv"
DRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "coupling" / "drainage_interval_routing.csv"
MASS_PATH = PROJECT_ROOT / "data" / "processed" / "coupling" / "interval_mass_balance.csv"
OVERFLOW_PATH = PROJECT_ROOT / "data" / "processed" / "coupling" / "drainage_overflow_points.geojson"
MAPPING_PATH = PROJECT_ROOT / "data" / "processed" / "coupling" / "inlet_surface_mapping.csv"
EPS = 1e-9


def main():
    print("\n=== RUNOFF-DRAINAGE COUPLING VALIDATION ===\n")
    errors = []

    for path in [INLETS_PATH, INLET_PATH, DRAIN_PATH, MASS_PATH, OVERFLOW_PATH, MAPPING_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    inlets = gpd.read_file(INLETS_PATH)
    inlet_df = pd.read_csv(INLET_PATH)
    drain_df = pd.read_csv(DRAIN_PATH)
    mass_df = pd.read_csv(MASS_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)

    eligible_count = int((inlets["coupling_eligible"].astype(int) == 1).sum())
    mapped_count = int((mapping_df["surface_mapping_selected"].astype(int) == 1).sum())
    interval_count = int(mass_df["interval_id"].nunique())
    expected_inlet_rows = mapped_count * interval_count

    print(f"Road-eligible inferred inlets : {eligible_count}")
    print(f"Raster-mapped inlet points    : {mapped_count}")
    print(f"Intervals                     : {interval_count}")
    print(f"Inlet coupling rows            : {len(inlet_df)}")
    print(f"Expected inlet rows            : {expected_inlet_rows}")

    if len(inlet_df) != expected_inlet_rows:
        errors.append("unexpected inlet/interval row count")

    mapping_duplicate_ids = int(
        mapping_df.duplicated(subset=["inlet_id"]).sum()
    )

    mapping_eligible_rows = len(mapping_df)

    selected_mapping = mapping_df[
        mapping_df["surface_mapping_selected"].astype(int) == 1
    ].copy()

    duplicate_selected_cells = int(
        selected_mapping.duplicated(
            subset=["surface_row", "surface_col"]
        ).sum()
    )

    allowed_status = {
        "mapped",
        "excluded_outside_runoff_aoi",
        "excluded_outside_grid",
        "excluded_invalid_geometry",
        "excluded_same_30m_cell",
    }

    bad_mapping_status = int(
        (~mapping_df["surface_mapping_status"].astype(str).isin(allowed_status)).sum()
    )

    source_eligible_ids = set(
        inlets.loc[
            inlets["coupling_eligible"].astype(int) == 1,
            "inlet_id",
        ].astype(str)
    )

    audited_ids = set(
        mapping_df["inlet_id"].astype(str)
    )

    audit_coverage_bad = 0 if source_eligible_ids == audited_ids else 1

    coupled_ids = set(
        inlet_df["inlet_id"].astype(str)
    )

    selected_ids = set(
        selected_mapping["inlet_id"].astype(str)
    )

    selected_id_mismatch = 0 if coupled_ids == selected_ids else 1

    print("\nSurface mapping audit")
    print(f"Audit rows                    : {mapping_eligible_rows}")
    print(f"Duplicate audit inlet IDs     : {mapping_duplicate_ids}")
    print(f"Duplicate selected cells      : {duplicate_selected_cells}")
    print(f"Bad mapping status rows       : {bad_mapping_status}")
    print(f"Audit covers all eligible IDs : {audit_coverage_bad == 0}")
    print(f"Coupled IDs match mapped IDs  : {selected_id_mismatch == 0}")

    if mapping_eligible_rows != eligible_count:
        errors.append("mapping audit row count differs from road-eligible inlet count")
    if mapping_duplicate_ids:
        errors.append("duplicate inlet IDs in mapping audit")
    if duplicate_selected_cells:
        errors.append("multiple selected inlets share a runoff cell")
    if bad_mapping_status:
        errors.append("invalid surface mapping status")
    if audit_coverage_bad:
        errors.append("mapping audit does not cover all road-eligible inlet IDs")
    if selected_id_mismatch:
        errors.append("coupled inlet IDs differ from raster-mapped inlet IDs")
    if mapped_count <= 0:
        errors.append("no raster-mapped inlet candidates")

    duplicate_rows = int(inlet_df.duplicated(subset=["interval_id", "inlet_id"]).sum())
    print(f"Duplicate inlet rows      : {duplicate_rows}")
    if duplicate_rows:
        errors.append("duplicate inlet interval rows")

    # Inlet checks
    inlet_numeric = ["duration_s", "arrival_volume_m3", "potential_inflow_q_m3s", "entry_capacity_m3s"]
    inlet_nonfinite = 0
    for field in inlet_numeric:
        values = inlet_df[field].astype(float)
        inlet_nonfinite += int((~np.isfinite(values)).sum())

    negative_arrivals = int((inlet_df["arrival_volume_m3"].astype(float) < -EPS).sum())
    q_mismatch = int((np.abs(
        inlet_df["potential_inflow_q_m3s"].astype(float) * inlet_df["duration_s"].astype(float)
        - inlet_df["arrival_volume_m3"].astype(float)
    ) > 1e-5).sum())
    observed_entry = int((inlet_df["entry_capacity_is_observed"].astype(int) != 0).sum())
    wrong_method = int((inlet_df["coupling_method"].astype(str) != "first_eligible_inlet_on_d8_path").sum())

    print("\nInlet coupling checks")
    print(f"Non-finite inlet values  : {inlet_nonfinite}")
    print(f"Negative arrivals         : {negative_arrivals}")
    print(f"Volume/Q mismatches       : {q_mismatch}")
    print(f"Capacity marked observed  : {observed_entry}")
    print(f"Wrong coupling method     : {wrong_method}")

    if inlet_nonfinite: errors.append("non-finite inlet values")
    if negative_arrivals: errors.append("negative inlet arrivals")
    if q_mismatch: errors.append("inlet volume/discharge conversion mismatch")
    if observed_entry: errors.append("synthetic entry capacity marked observed")
    if wrong_method: errors.append("unexpected coupling method")

    # Drainage routing checks
    flow_fields = [
        "local_inlet_q_m3s", "upstream_q_m3s", "total_q_m3s",
        "conveyed_q_m3s", "overflow_q_m3s", "export_q_m3s", "overflow_volume_m3",
    ]
    drain_nonfinite = 0
    for field in flow_fields:
        values = drain_df[field].astype(float)
        drain_nonfinite += int((~np.isfinite(values)).sum())

    negative_rows = int((drain_df[flow_fields].astype(float) < -EPS).any(axis=1).sum())
    node_balance_bad = int((np.abs(
        drain_df["total_q_m3s"].astype(float)
        - drain_df["conveyed_q_m3s"].astype(float)
        - drain_df["overflow_q_m3s"].astype(float)
        - drain_df["export_q_m3s"].astype(float)
    ) > 1e-8).sum())

    non_sink = drain_df[drain_df["outgoing_drain_id"].fillna("").astype(str) != ""].copy()
    over_capacity = int(((
        non_sink["conveyed_q_m3s"].astype(float)
        - non_sink["capacity_m3s"].astype(float)
    ) > 1e-8).sum())
    observed_route = int((non_sink["capacity_is_observed"].astype(float) != 0).sum())
    overflow_volume_bad = int((np.abs(
        drain_df["overflow_q_m3s"].astype(float) * drain_df["duration_s"].astype(float)
        - drain_df["overflow_volume_m3"].astype(float)
    ) > 1e-5).sum())

    print("\nDrainage routing checks")
    print(f"Non-finite flow values   : {drain_nonfinite}")
    print(f"Negative flow rows        : {negative_rows}")
    print(f"Node flow mismatches      : {node_balance_bad}")
    print(f"Conveyed over capacity    : {over_capacity}")
    print(f"Capacity marked observed  : {observed_route}")
    print(f"Overflow volume mismatch  : {overflow_volume_bad}")

    if drain_nonfinite: errors.append("non-finite drainage values")
    if negative_rows: errors.append("negative drainage flows")
    if node_balance_bad: errors.append("drainage node flow mismatch")
    if over_capacity: errors.append("conveyed flow exceeds capacity")
    if observed_route: errors.append("synthetic routing capacity marked observed")
    if overflow_volume_bad: errors.append("overflow discharge/volume mismatch")

    # Interval mass balance
    surface_bad = 0
    drain_bad = 0
    capture_input_bad = 0
    for row in mass_df.itertuples(index=False):
        surface_error = abs(
            float(row.generated_surface_volume_m3)
            - float(row.captured_to_drain_volume_m3)
            - float(row.surface_export_volume_m3)
        )
        if surface_error > max(0.05, float(row.generated_surface_volume_m3) * 1e-8):
            surface_bad += 1

        drain_error = abs(
            float(row.drainage_input_volume_m3)
            - float(row.drainage_export_volume_m3)
            - float(row.drainage_overflow_volume_m3)
        )
        if drain_error > max(0.05, float(row.drainage_input_volume_m3) * 1e-8):
            drain_bad += 1

        capture_error = abs(float(row.captured_to_drain_volume_m3) - float(row.drainage_input_volume_m3))
        if capture_error > max(0.05, float(row.captured_to_drain_volume_m3) * 1e-8):
            capture_input_bad += 1

    print("\nInterval mass-balance checks")
    print(f"Surface balance failures : {surface_bad}")
    print(f"Drain balance failures   : {drain_bad}")
    print(f"Capture/input mismatches : {capture_input_bad}")

    if surface_bad: errors.append("surface mass balance failure")
    if drain_bad: errors.append("drainage mass balance failure")
    if capture_input_bad: errors.append("surface capture differs from drainage input")

    # Overflow output count
    try:
        overflow = gpd.read_file(OVERFLOW_PATH)
        overflow_count = len(overflow)
    except Exception:
        overflow_count = 0
    positive_overflow_rows = int((drain_df["overflow_volume_m3"].astype(float) > EPS).sum())

    print("\nOverflow output")
    print(f"Positive overflow rows   : {positive_overflow_rows}")
    print(f"Overflow GeoJSON features: {overflow_count}")
    if positive_overflow_rows != overflow_count:
        errors.append("overflow GeoJSON feature count mismatch")

    print("\nInterval summary")
    for row in mass_df.itertuples(index=False):
        print(
            f"{int(row.start_min):>3}-{int(row.end_min):>3} min | "
            f"generated {float(row.generated_surface_volume_m3):>10.2f} m3 | "
            f"captured {float(row.captured_to_drain_volume_m3):>10.2f} m3 | "
            f"drain overflow {float(row.drainage_overflow_volume_m3):>10.2f} m3 | "
            f"drain export {float(row.drainage_export_volume_m3):>10.2f} m3"
        )

    print("\nIMPORTANT:")
    print("Coupling points are inferred, not surveyed municipal inlet locations.")
    print("Drain capacities are synthetic prototype Manning-scenario capacities.")
    print("\n====================================")
    if errors:
        print("RUNOFF-DRAINAGE COUPLING VALIDATION: FAILED")
        for error in errors:
            print(f" - {error}")
    else:
        print("RUNOFF-DRAINAGE COUPLING VALIDATION: PASSED")
    print("====================================\n")


if __name__ == "__main__":
    main()
