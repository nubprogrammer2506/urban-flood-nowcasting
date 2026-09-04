import json
from pathlib import Path

RAIN_FILE = Path("data/raw/rainfall/heavy_rain_demo.json")


def main():
    if not RAIN_FILE.exists():
        raise FileNotFoundError(f"Rainfall file not found: {RAIN_FILE}")

    with open(RAIN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_top_level = [
        "scenario_id",
        "location",
        "source",
        "interval_minutes",
        "forecast_horizon_minutes",
        "forecast",
    ]

    for key in required_top_level:
        if key not in data:
            raise ValueError(f"Missing required field: {key}")

    forecast = data["forecast"]

    if not isinstance(forecast, list) or len(forecast) == 0:
        raise ValueError("forecast must be a non-empty list")

    previous_minutes = -1

    for i, step in enumerate(forecast):
        if "minutes" not in step:
            raise ValueError(f"Step {i} missing 'minutes'")

        if "rainfall_mm_hr" not in step:
            raise ValueError(f"Step {i} missing 'rainfall_mm_hr'")

        minutes = step["minutes"]
        rainfall = step["rainfall_mm_hr"]

        if minutes < 0:
            raise ValueError(f"Step {i}: minutes cannot be negative")

        if rainfall < 0:
            raise ValueError(f"Step {i}: rainfall cannot be negative")

        if minutes <= previous_minutes:
            raise ValueError("Forecast minutes must be strictly increasing")

        previous_minutes = minutes

    interval = data["interval_minutes"]
    horizon = data["forecast_horizon_minutes"]

    if forecast[-1]["minutes"] != horizon:
        raise ValueError(
            f"Final forecast step must equal horizon: "
            f"{forecast[-1]['minutes']} != {horizon}"
        )

    expected_steps = horizon // interval + 1

    if len(forecast) != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} forecast steps, "
            f"found {len(forecast)}"
        )

    print("Rainfall scenario valid")
    print(f"Scenario: {data['scenario_id']}")
    print(f"Location: {data['location']}")
    print(f"Source: {data['source']}")
    print(f"Interval: {interval} minutes")
    print(f"Horizon: {horizon} minutes")
    print(f"Forecast steps: {len(forecast)}")

    print("\nRainfall timeline:")

    for step in forecast:
        print(
            f"+{step['minutes']:>3} min -> "
            f"{step['rainfall_mm_hr']:>5.1f} mm/hr"
        )


if __name__ == "__main__":
    main()