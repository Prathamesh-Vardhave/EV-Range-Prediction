from pydantic import BaseModel, Field


class EVFeatures(BaseModel):
    """Raw input features required to predict real-world EV range."""

    make: str = Field(..., example="Tata", description="Vehicle manufacturer")
    battery_capacity_kWh: float = Field(..., example=35.3, description="Current usable battery capacity (kWh)")
    battery_capacity_kWh_new: float = Field(..., example=40.5, description="Original (as-new) battery capacity (kWh)")
    motor_power_kW: float = Field(..., example=105, description="Motor power output (kW)")
    vehicle_weight_kg: float = Field(..., example=1610, description="Total vehicle weight (kg)")
    drag_coefficient_Cd: float = Field(..., example=0.34, description="Aerodynamic drag coefficient")
    frontal_area_m2: float = Field(..., example=2.4, description="Frontal area (m^2)")
    rolling_resistance_Cr: float = Field(..., example=0.0122, description="Rolling resistance coefficient")
    motor_efficiency_pct: float = Field(..., example=91, description="Motor efficiency (%)")
    avg_driving_speed_kmh: float = Field(..., example=27.0, description="Average driving speed (km/h)")
    ambient_temperature_C: float = Field(..., example=32.5, description="Ambient temperature (Celsius)")
    ac_usage: int = Field(..., ge=0, le=1, example=1, description="AC usage: 0=off, 1=on")
    terrain_type: int = Field(..., ge=0, le=2, example=0, description="Terrain type code (0/1/2)")
    traffic_condition: int = Field(..., ge=0, le=2, example=1, description="Traffic condition code (0/1/2)")
    driving_style: int = Field(..., ge=0, le=2, example=0, description="Driving style code (0=calm ... 2=aggressive)")
    passengers: int = Field(..., ge=1, example=1, description="Number of passengers")
    battery_age_years: float = Field(..., example=1.5, description="Battery age in years")
    tyre_pressure_deviation_psi: int = Field(..., example=0, description="Deviation from recommended tyre pressure (psi)")
    city_road_type: str = Field(..., example="Delhi", description="City / road type category")
    price_lakh_INR: float = Field(..., example=17.5, description="Vehicle price (INR lakh)")

    class Config:
        schema_extra = {
            "example": {
                "make": "Tata",
                "battery_capacity_kWh": 35.3,
                "battery_capacity_kWh_new": 40.5,
                "motor_power_kW": 105,
                "vehicle_weight_kg": 1610,
                "drag_coefficient_Cd": 0.34,
                "frontal_area_m2": 2.4,
                "rolling_resistance_Cr": 0.0122,
                "motor_efficiency_pct": 91,
                "avg_driving_speed_kmh": 27.0,
                "ambient_temperature_C": 32.5,
                "ac_usage": 1,
                "terrain_type": 0,
                "traffic_condition": 1,
                "driving_style": 0,
                "passengers": 1,
                "battery_age_years": 1.5,
                "tyre_pressure_deviation_psi": 0,
                "city_road_type": "Delhi",
                "price_lakh_INR": 17.5,
            }
        }


class RangePrediction(BaseModel):
    """Response payload returned by the /predict endpoint."""

    predicted_range_km: float = Field(..., description="Predicted real-world EV range in kilometers")
