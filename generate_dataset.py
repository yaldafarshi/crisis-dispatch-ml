"""
Synthetic dataset generator for the crisis-dispatch project.

Simulates 2,000 electrical-network fault incidents in the Tehran area using
physics- and rule-based logic (Haversine travel distance, seasonal/hourly
weather, per-team skill matrices, workload-based availability, etc.).

Outputs:
    crisis_dataset_regression_v52.csv   -> for repair-time regression
    crisis_dataset_classifier_v52.csv   -> for optimal-team classification

Run:
    python data_generation/generate_dataset.py
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# 1. Helper functions
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Great-circle (Haversine) distance between two coordinates, in kilometers.
    """
    R = 6371  # Earth radius, km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * R


def calculate_dynamic_weather(day_of_year, hour):
    """
    Generates temperature and humidity as a function of season and hour of
    day, with random noise layered on top for realism.
    """
    season_idx = int((day_of_year // 91) % 4)
    base_temps = [15, 32, 18, 5]  # average temperature per season
    base_temp = base_temps[season_idx]

    # daily (sinusoidal) temperature variation
    daily_variation = 8 * np.cos((hour - 14) * 2 * np.pi / 24)
    noise_temp = np.random.normal(0, 3)
    temperature = base_temp + daily_variation + noise_temp

    base_hum = [55, 35, 50, 65][season_idx]
    humidity_daily = -5 * np.cos((hour - 4) * 2 * np.pi / 24)
    noise_hum = np.random.normal(0, 8)
    humidity = np.clip(base_hum + humidity_daily + noise_hum, 10, 95)

    return temperature, humidity


def generate_skill_matrix(n_teams, n_fault_types):
    """
    Builds a team/skill matrix. Each team gets one primary specialty
    (skill level 2) and one or two secondary specialties (skill level 1).
    """
    matrix = np.zeros((n_teams, n_fault_types), dtype=int)
    for t in range(n_teams):
        main_skill = np.random.randint(0, n_fault_types)
        matrix[t, main_skill] = 2

        others = [i for i in range(n_fault_types) if i != main_skill]
        num_secondary = np.random.choice([1, 2], p=[0.7, 0.3])
        secondary_skills = np.random.choice(others, size=num_secondary, replace=False)
        for s in secondary_skills:
            matrix[t, s] = 1

    return matrix


# ==============================================================================
# 2. Time / scoring logic
# ==============================================================================

def calculate_realistic_times(dist_km, fault_type, skill_level, equipment_age,
                               weather_condition, humidity, workload, road_quality, is_noise=True):
    """
    Computes travel time and repair time from physical and human factors.
    """
    # --- Travel time ---
    base_speed = 35  # km/h
    if road_quality <= 2:
        base_speed *= 0.7  # slower on poor-quality roads
    if weather_condition == 'storm':
        base_speed *= 0.5
    elif weather_condition == 'rain':
        base_speed *= 0.8

    travel_time = (dist_km / base_speed) * 60  # minutes

    # --- Repair time ---
    base_times = {0: 30, 1: 60, 2: 120, 3: 45, 4: 20}
    base_time = base_times.get(fault_type, 60)

    aging_factor = 1 + (0.015 * equipment_age)
    inefficiency = {2: 0.05, 1: 0.15, 0: 0.40}[skill_level]
    weather_factor = {'good': 1.0, 'rain': 1.2, 'fog': 1.3, 'storm': 1.8}[weather_condition]

    # humidity raises risk of shock / sensitive-equipment failure
    humidity_factor = 1.0
    if humidity > 80:
        humidity_factor = 1.1

    estimated_repair = base_time * aging_factor * (1 + inefficiency) * weather_factor * humidity_factor

    # --- Random noise ---
    if is_noise:
        # simulate real-world uncertainty (80% to 140% of the base estimate)
        noise_multiplier = np.random.uniform(0.8, 1.4)
        actual_repair = max(5, estimated_repair * noise_multiplier)
    else:
        actual_repair = estimated_repair

    return travel_time, actual_repair


# ==============================================================================
# 3. Main dataset generation function
# ==============================================================================

def generate_crisis_dataset(n_samples=2000, n_teams=5, n_fault_types=5):
    # Team depot locations (hypothetical, spread across Tehran)
    team_locations = [
        {'id': 1, 'lat': 35.78, 'lon': 51.42},  # North Tehran
        {'id': 2, 'lat': 35.73, 'lon': 51.40},  # Central Tehran
        {'id': 3, 'lat': 35.71, 'lon': 51.35},  # East Tehran
        {'id': 4, 'lat': 35.72, 'lon': 51.45},  # West Tehran
        {'id': 5, 'lat': 35.67, 'lon': 51.39},  # South Tehran
    ]

    skill_matrix = generate_skill_matrix(n_teams, n_fault_types)

    # base success rate per team (used for the success_rate feature)
    team_base_rates = {t['id']: 0.6 + np.random.uniform(0, 0.3) for t in team_locations}

    # spatial clusters where faults tend to occur, instead of one central point
    failure_clusters = np.array([
        [35.75, 51.42],
        [35.70, 51.38],
        [35.73, 51.45],
        [35.68, 51.40]
    ])

    data = []

    for i in range(n_samples):
        # --- fault location ---
        cluster_idx = np.random.choice(len(failure_clusters))
        center_lat, center_lon = failure_clusters[cluster_idx]

        # spread the fault around the cluster center (std dev 0.02 degrees)
        lat_failure = np.random.normal(center_lat, 0.02)
        lon_failure = np.random.normal(center_lon, 0.02)

        # --- time features ---
        hour = np.random.randint(0, 24)
        hour_sin, hour_cos = np.sin(hour * 2 * np.pi / 24), np.cos(hour * 2 * np.pi / 24)

        day = np.random.randint(1, 366)
        day_sin, day_cos = np.sin(day * 2 * np.pi / 365), np.cos(day * 2 * np.pi / 365)
        day_of_week = (day - 1) % 7  # 0: Saturday ... 6: Friday

        season_idx = int((day // 91) % 4)
        season_onehot = [0] * 4
        season_onehot[season_idx] = 1
        dow_onehot = [0] * 7
        dow_onehot[day_of_week] = 1

        # --- environmental features ---
        temperature, humidity = calculate_dynamic_weather(day, hour)
        load_kw = np.random.uniform(500, 1500)

        # --- fault features ---
        fault_type = np.random.randint(0, n_fault_types)
        fault_onehot = [0] * n_fault_types
        fault_onehot[fault_type] = 1

        traffic_level = np.random.randint(1, 6)
        road_quality = np.random.randint(1, 6)
        weather_condition = np.random.choice(['good', 'rain', 'fog', 'storm'], p=[0.6, 0.2, 0.1, 0.1])
        equipment_age = np.random.randint(1, 25)

        # --- engineered / interaction features ---
        thermal_age_interaction = temperature * equipment_age
        weather_severity = {'good': 0, 'rain': 1, 'fog': 2, 'storm': 3}.get(weather_condition, 0)
        complexity = {0: 1, 1: 2, 2: 3, 3: 2, 4: 1}.get(fault_type, 1)

        # crisis score: blend of fault complexity, weather, and traffic
        crisis_score = (complexity * 2) + (weather_severity * 1.5) + (traffic_level / 2)

        # equipment risk index
        equipment_risk_index = (equipment_age / 25) * (temperature / 40) * (load_kw / 1000)

        # --- per-team features ---
        team_metrics = []
        available_teams_info = []

        for t in team_locations:
            t_id = t['id']
            t_idx = t_id - 1

            dist_km = haversine_vectorized(lat_failure, lon_failure, t['lat'], t['lon'])
            skill_level = skill_matrix[t_idx, fault_type]

            # current workload (0-5); a team is available only if workload <= 3
            current_workload = np.random.randint(0, 6)
            is_available = 1 if current_workload <= 3 else 0

            travel_time, repair_time = calculate_realistic_times(
                dist_km, fault_type, skill_level, equipment_age,
                weather_condition, humidity, current_workload, road_quality, is_noise=True
            )
            total_time = travel_time + repair_time

            # score used to pick the team (lower is better)
            inefficiency_cost = {2: 5, 1: 15, 0: 30}[skill_level]
            weather_penalty = {'good': 0, 'rain': 10, 'fog': 20, 'storm': 50}[weather_condition]

            score = (dist_km * 2) + inefficiency_cost + weather_penalty + (current_workload * 5)

            team_metrics.append({
                'id': t_id,
                'dist_km': dist_km,
                'skill_level': skill_level,
                'score': score,
                'total_time': total_time,
                'workload': current_workload,
                'success_rate': team_base_rates[t_id]
            })

            if is_available:
                available_teams_info.append(t_id)

        # --- pick the dispatched team ---
        if available_teams_info:
            # 90% of the time: the lowest-score (optimal) available team is picked
            # 10% of the time: a random available team is picked (human/system error)
            if np.random.random() < 0.9:
                available_metrics = [tm for tm in team_metrics if tm['id'] in available_teams_info]
                best_team = min(available_metrics, key=lambda x: x['score'])
                selected_team_id = best_team['id']
                target_total_time = best_team['total_time']
            else:
                random_team_id = np.random.choice(available_teams_info)
                random_team_data = next(tm for tm in team_metrics if tm['id'] == random_team_id)
                selected_team_id = random_team_id
                target_total_time = random_team_data['total_time']
        else:
            # no team available for this incident
            selected_team_id = -1
            target_total_time = -1

        row = {
            'incident_id': i,
            'lat_failure': lat_failure,
            'lon_failure': lon_failure,
            'hour_sin': hour_sin, 'hour_cos': hour_cos,
            'day_sin': day_sin, 'day_cos': day_cos,
            'season_1': season_onehot[0], 'season_2': season_onehot[1],
            'season_3': season_onehot[2], 'season_4': season_onehot[3],
            'day_of_week_0': dow_onehot[0], 'day_of_week_1': dow_onehot[1],
            'day_of_week_2': dow_onehot[2], 'day_of_week_3': dow_onehot[3],
            'day_of_week_4': dow_onehot[4], 'day_of_week_5': dow_onehot[5],
            'day_of_week_6': dow_onehot[6],
            'temperature': temperature,
            'humidity': humidity,
            'load_kW': load_kw,
            'voltage_drop': np.random.uniform(30, 70),
            'thermal_stress': temperature * humidity / 1000,
            'traffic_level': traffic_level,
            'road_quality': road_quality,
            'weather_condition': weather_condition,

            'fault_type_1': fault_onehot[0], 'fault_type_2': fault_onehot[1],
            'fault_type_3': fault_onehot[2], 'fault_type_4': fault_onehot[3],
            'fault_type_5': fault_onehot[4],

            'equipment_age': equipment_age,
            'thermal_age_interaction': thermal_age_interaction,
            'equipment_risk_index': equipment_risk_index,
            'crisis_score': crisis_score,

            'team_1_dist': team_metrics[0]['dist_km'],
            'team_2_dist': team_metrics[1]['dist_km'],
            'team_3_dist': team_metrics[2]['dist_km'],
            'team_4_dist': team_metrics[3]['dist_km'],
            'team_5_dist': team_metrics[4]['dist_km'],

            'team_1_skill': team_metrics[0]['skill_level'],
            'team_2_skill': team_metrics[1]['skill_level'],
            'team_3_skill': team_metrics[2]['skill_level'],
            'team_4_skill': team_metrics[3]['skill_level'],
            'team_5_skill': team_metrics[4]['skill_level'],

            'team_1_workload': team_metrics[0]['workload'],
            'team_2_workload': team_metrics[1]['workload'],
            'team_3_workload': team_metrics[2]['workload'],
            'team_4_workload': team_metrics[3]['workload'],
            'team_5_workload': team_metrics[4]['workload'],

            'team_1_success_rate': team_metrics[0]['success_rate'],
            'team_2_success_rate': team_metrics[1]['success_rate'],
            'team_3_success_rate': team_metrics[2]['success_rate'],
            'team_4_success_rate': team_metrics[3]['success_rate'],
            'team_5_success_rate': team_metrics[4]['success_rate'],

            'target_team_id': selected_team_id,
            'target_total_time': target_total_time
        }
        data.append(row)

    df = pd.DataFrame(data)
    # drop incidents where no team was available at all
    df_clean = df[df['target_team_id'] != -1].copy()
    return df_clean


# ==============================================================================
# 4. Run and save
# ==============================================================================

if __name__ == "__main__":
    print("Generating crisis-dispatch dataset...")

    df_all = generate_crisis_dataset(n_samples=2000, n_teams=5, n_fault_types=5)

    # --- regression dataset: predict total repair time ---
    df_regression = df_all.drop(columns=['target_team_id'])
    df_regression.to_csv('crisis_dataset_regression_v52.csv', index=False)
    print(f"Saved regression dataset: {len(df_regression)} samples")

    # --- classification dataset: predict the optimal team id ---
    df_classifier = df_all.drop(columns=['target_total_time'])
    df_classifier.to_csv('crisis_dataset_classifier_v52.csv', index=False)
    print(f"Saved classification dataset: {len(df_classifier)} samples")

    print("\nSummary statistics:")
    print(df_all.describe())

    print("\nTarget team distribution:")
    print(df_all['target_team_id'].value_counts().sort_index())

    print("\nDone.")
