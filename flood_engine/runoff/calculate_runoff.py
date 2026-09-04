import json
from pathlib import Path


RAIN_FILE = Path("data/raw/rainfall/heavy_rain_demo.json")

RUNOFF_COEFFICIENT = 0.80

CELL_SIZE_M = 30
CELL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M


def main():
    with open(RAIN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    forecast = data["forecast"]

    total_rainfall_mm = 0.0
    total_runoff_mm = 0.0
    total_runoff_volume_m3 = 0.0

    print("Rainfall -> Runoff calculation")
    print(f"Runoff coefficient: {RUNOFF_COEFFICIENT}")
    print(f"Terrain cell area: {CELL_AREA_M2} m²")
    print()

    for i in range(len(forecast) - 1):
        current = forecast[i]
        next_step = forecast[i + 1]

        start_min = current["minutes"]
        end_min = next_step["minutes"]

        duration_minutes = end_min - start_min
        duration_hours = duration_minutes / 60.0

        start_intensity = current["rainfall_mm_hr"]
        end_intensity = next_step["rainfall_mm_hr"]

        average_intensity = (
            start_intensity + end_intensity
        ) / 2.0

        rainfall_mm = average_intensity * duration_hours

        runoff_mm = rainfall_mm * RUNOFF_COEFFICIENT

        runoff_depth_m = runoff_mm / 1000.0

        runoff_volume_m3 = runoff_depth_m * CELL_AREA_M2

        total_rainfall_mm += rainfall_mm
        total_runoff_mm += runoff_mm
        total_runoff_volume_m3 += runoff_volume_m3

        print(
            f"{start_min:>3}-{end_min:>3} min | "
            f"Rain = {rainfall_mm:>6.2f} mm | "
            f"Runoff = {runoff_mm:>6.2f} mm | "
            f"Volume/cell = {runoff_volume_m3:>6.2f} m³"
        )

    print()
    print("Summary")
    print(f"Total rainfall: {total_rainfall_mm:.2f} mm")
    print(f"Total effective runoff: {total_runoff_mm:.2f} mm")
    print(
        f"Total runoff volume per 30 m cell: "
        f"{total_runoff_volume_m3:.2f} m³"
    )


if __name__ == "__main__":
    main()