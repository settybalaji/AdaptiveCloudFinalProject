# ================================================================
# MCA FINAL YEAR PROJECT
# ADAPTIVE CLOUD RESOURCE ALLOCATION USING MACHINE LEARNING
# AND COST-AWARE WORKLOAD PREDICTION
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    IsolationForest
)

from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

np.random.seed(42)

# ================================================================
# 1. PROJECT CONFIGURATION
# ================================================================

PROJECT_NAME = "Adaptive Cloud Resource Allocation Using Machine Learning and Cost-Aware Workload Prediction"

OUTPUT_FOLDER = "cloud_project_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CPU_RATE = 0.040
MEMORY_RATE = 0.005

CPU_CAPACITY_PER_CORE = 25
MEMORY_CAPACITY_PER_GB = 10

SAFETY_MARGIN = 1.15

CPU_UTILIZATION_TARGET = 0.75
MEMORY_UTILIZATION_TARGET = 0.80

MAX_CPU_CORES = 16
MAX_MEMORY_GB = 32


# ================================================================
# 2. CREATE SAMPLE CLOUD WORKLOAD DATASET
# ================================================================

def create_sample_dataset(rows=3000):

    rng = np.random.default_rng(42)

    t = np.arange(rows)

    hour = t % 24
    day = t // 24

    daily_pattern = 13 * np.sin(
        2 * np.pi * (hour - 7) / 24
    )

    weekly_pattern = 7 * np.sin(
        2 * np.pi * day / 7
    )

    trend = 0.003 * t

    cpu = 43 + daily_pattern + weekly_pattern + trend

    cpu += rng.normal(0, 5.5, rows)

    burst = rng.random(rows) < 0.035

    cpu[burst] += rng.uniform(
        12, 28, burst.sum()
    )

    cpu = np.clip(cpu, 5, 98)

    memory = 24 + 0.62 * cpu

    memory += 4 * np.sin(
        2 * np.pi * (hour - 3) / 24
    )

    memory += rng.normal(0, 4, rows)

    memory = np.clip(memory, 8, 98)

    return pd.DataFrame({

        "timestamp": pd.date_range(
            "2025-01-01",
            periods=rows,
            freq="h"
        ),

        "cpu_usage": cpu,

        "memory_usage": memory,

        "request_count": np.maximum(
            20,
            (cpu * 18 + rng.normal(0, 100, rows)).astype(int)
        )

    })


# ================================================================
# 3. LOAD DATASET
# ================================================================

USE_SAMPLE_DATA = True

CSV_PATH = "cloud_workload.csv"

if USE_SAMPLE_DATA:

    data = create_sample_dataset()

else:

    data = pd.read_csv(CSV_PATH)


required_columns = [
    "timestamp",
    "cpu_usage",
    "memory_usage"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce"
)

data["cpu_usage"] = pd.to_numeric(
    data["cpu_usage"],
    errors="coerce"
)

data["memory_usage"] = pd.to_numeric(
    data["memory_usage"],
    errors="coerce"
)

data = data.dropna(
    subset=required_columns
)

data = data.sort_values(
    "timestamp"
).drop_duplicates(
    subset=["timestamp"]
).reset_index(drop=True)

data["cpu_usage"] = data["cpu_usage"].clip(0, 100)

data["memory_usage"] = data["memory_usage"].clip(0, 100)

if len(data) < 100:

    raise ValueError(
        "At least 100 valid workload records are required."
    )

print("DATASET LOADED SUCCESSFULLY")

print(data.head())

print("Total Records:", len(data))


# ================================================================
# 4. DATA VISUALIZATION
# ================================================================

plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["cpu_usage"]
)

plt.title("Cloud CPU Usage")

plt.xlabel("Timestamp")

plt.ylabel("CPU Usage (%)")

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/cpu_history.png"
)

plt.show()

print("STEP 4 COMPLETED: CPU GRAPH SAVED")
plt.figure(figsize=(12, 5))

plt.plot(
    data["timestamp"],
    data["memory_usage"]
)

plt.title("Cloud Memory Usage")

plt.xlabel("Timestamp")

plt.ylabel("Memory Usage (%)")

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/memory_history.png"
)

plt.show()


# ================================================================
# 5. ANOMALY DETECTION
# ================================================================

anomaly_model = IsolationForest(

    n_estimators=150,

    contamination=0.03,

    random_state=42

)

anomaly_result = anomaly_model.fit_predict(

    data[
        ["cpu_usage", "memory_usage"]
    ]

)

data["is_anomaly"] = anomaly_result == -1

print("\nANOMALY DETECTION RESULTS")

print(
    "Detected Anomalies:",
    data["is_anomaly"].sum()
)


plt.figure(figsize=(9, 5))

normal = data[data["is_anomaly"] == False]

abnormal = data[data["is_anomaly"] == True]

plt.scatter(
    normal["cpu_usage"],
    normal["memory_usage"],
    label="Normal"
)

plt.scatter(
    abnormal["cpu_usage"],
    abnormal["memory_usage"],
    label="Anomaly"
)

plt.xlabel("CPU Usage")

plt.ylabel("Memory Usage")

plt.title("Cloud Workload Anomaly Detection")

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/anomaly_detection.png"
)

plt.show()
print("STEP 4 COMPLETED: MEMORY GRAPH SAVED")

# ================================================================
# 6. FEATURE ENGINEERING
# ================================================================

for lag in [1, 2, 3, 6, 12, 24]:

    data[f"cpu_lag_{lag}"] = (
        data["cpu_usage"].shift(lag)
    )

    data[f"memory_lag_{lag}"] = (
        data["memory_usage"].shift(lag)
    )


data["cpu_rolling_mean"] = (
    data["cpu_usage"].shift(1).rolling(6).mean()
)

data["memory_rolling_mean"] = (
    data["memory_usage"].shift(1).rolling(6).mean()
)

data["cpu_rolling_std"] = (
    data["cpu_usage"].shift(1).rolling(6).std()
)

data["memory_rolling_std"] = (
    data["memory_usage"].shift(1).rolling(6).std()
)


data["hour"] = data["timestamp"].dt.hour

data["day"] = data["timestamp"].dt.dayofweek

data["is_weekend"] = (
    data["day"] >= 5
).astype(int)


data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)


data["target_cpu"] = data["cpu_usage"].shift(-1)

data["target_memory"] = data["memory_usage"].shift(-1)


data = data.dropna().reset_index(drop=True)


features = [

    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "cpu_lag_6",
    "cpu_lag_12",
    "cpu_lag_24",

    "memory_lag_1",
    "memory_lag_2",
    "memory_lag_3",
    "memory_lag_6",
    "memory_lag_12",
    "memory_lag_24",

    "cpu_rolling_mean",
    "memory_rolling_mean",

    "cpu_rolling_std",
    "memory_rolling_std",

    "hour",
    "day",
    "is_weekend",

    "hour_sin",
    "hour_cos"

]


X = data[features]

y = data[
    ["target_cpu", "target_memory"]
]


# ================================================================
# 7. TRAINING AND TESTING SPLIT
# ================================================================

split = int(len(data) * 0.80)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]

print("\nTRAINING RECORDS:", len(X_train))

print("TESTING RECORDS:", len(X_test))


# ================================================================
# 8. RANDOM FOREST MODEL
# ================================================================

random_forest = MultiOutputRegressor(

    RandomForestRegressor(

        n_estimators=180,

        max_depth=14,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1

    )

)

random_forest.fit(
    X_train,
    y_train
)


rf_prediction = random_forest.predict(
    X_test
)


# ================================================================
# 9. EXTRA TREES MODEL
# ================================================================

extra_trees = MultiOutputRegressor(

    ExtraTreesRegressor(

        n_estimators=180,

        max_depth=14,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1

    )

)

extra_trees.fit(
    X_train,
    y_train
)


et_prediction = extra_trees.predict(
    X_test
)


# ================================================================
# 10. MODEL EVALUATION
# ================================================================

def evaluate_model(actual, predicted):

    return {

        "MAE": mean_absolute_error(
            actual,
            predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "R2": r2_score(
            actual,
            predicted
        )

    }


rf_cpu_metrics = evaluate_model(
    y_test["target_cpu"],
    rf_prediction[:, 0]
)

et_cpu_metrics = evaluate_model(
    y_test["target_cpu"],
    et_prediction[:, 0]
)

rf_memory_metrics = evaluate_model(
    y_test["target_memory"],
    rf_prediction[:, 1]
)

et_memory_metrics = evaluate_model(
    y_test["target_memory"],
    et_prediction[:, 1]
)


model_comparison = pd.DataFrame([

    {
        "Model": "Random Forest",
        "CPU MAE": rf_cpu_metrics["MAE"],
        "Memory MAE": rf_memory_metrics["MAE"]
    },

    {
        "Model": "Extra Trees",
        "CPU MAE": et_cpu_metrics["MAE"],
        "Memory MAE": et_memory_metrics["MAE"]
    }

])


print("\nMODEL COMPARISON")

print(model_comparison.round(3))


# Select the model with the lowest average prediction MAE.

rf_average_mae = (
    rf_cpu_metrics["MAE"]
    + rf_memory_metrics["MAE"]
) / 2

et_average_mae = (
    et_cpu_metrics["MAE"]
    + et_memory_metrics["MAE"]
) / 2


if rf_average_mae <= et_average_mae:

    selected_model = random_forest

    predictions = rf_prediction

    selected_model_name = "Random Forest"

else:

    selected_model = extra_trees

    predictions = et_prediction

    selected_model_name = "Extra Trees"


print("\nSELECTED MODEL:", selected_model_name)


# ================================================================
# 11. PREDICTED CPU AND MEMORY
# ================================================================

predicted_cpu = np.clip(
    predictions[:, 0],
    0,
    100
)

predicted_memory = np.clip(
    predictions[:, 1],
    0,
    100
)


actual_cpu = y_test[
    "target_cpu"
].to_numpy()

actual_memory = y_test[
    "target_memory"
].to_numpy()


print("\nFINAL PREDICTION METRICS")

print(
    "CPU:",
    evaluate_model(actual_cpu, predicted_cpu)
)

print(
    "MEMORY:",
    evaluate_model(actual_memory, predicted_memory)
)


# ================================================================
# 12. WORKLOAD CLASSIFICATION
# ================================================================

def classify_workload(cpu, memory):

    peak = max(cpu, memory)

    if peak >= 85:

        return "CRITICAL"

    elif peak >= 65:

        return "HIGH"

    elif peak >= 40:

        return "MEDIUM"

    else:

        return "LOW"


workload_class = [

    classify_workload(cpu, memory)

    for cpu, memory in zip(
        predicted_cpu,
        predicted_memory
    )

]


# ================================================================
# 13. COST-AWARE RESOURCE OPTIMIZATION
# ================================================================

def calculate_cost(cores, memory_gb):

    return (

        cores * CPU_RATE

        + memory_gb * MEMORY_RATE

    )


def allocate_resources(cpu, memory):

    required_cpu = cpu * SAFETY_MARGIN

    required_memory = memory * SAFETY_MARGIN

    candidates = []

    for cores in range(1, MAX_CPU_CORES + 1):

        for gb in range(1, MAX_MEMORY_GB + 1):

            cpu_capacity = (
                cores
                * CPU_CAPACITY_PER_CORE
                * CPU_UTILIZATION_TARGET
            )

            memory_capacity = (
                gb
                * MEMORY_CAPACITY_PER_GB
                * MEMORY_UTILIZATION_TARGET
            )

            if (

                cpu_capacity >= required_cpu

                and memory_capacity >= required_memory

            ):

                cost = calculate_cost(
                    cores,
                    gb
                )

                candidates.append(
                    (cost, cores, gb)
                )

    if candidates:

        cost, cores, gb = min(candidates)

        status = "FEASIBLE"

    else:

        cores = MAX_CPU_CORES

        gb = MAX_MEMORY_GB

        cost = calculate_cost(
            cores,
            gb
        )

        status = "CAPACITY_LIMIT"

    return {

        "cpu_cores": cores,

        "memory_gb": gb,

        "cost": cost,

        "status": status

    }


adaptive_allocations = [

    allocate_resources(cpu, memory)

    for cpu, memory in zip(
        predicted_cpu,
        predicted_memory
    )

]


adaptive_cores = np.array([
    x["cpu_cores"] for x in adaptive_allocations
])

adaptive_memory = np.array([
    x["memory_gb"] for x in adaptive_allocations
])

adaptive_cost = np.array([
    x["cost"] for x in adaptive_allocations
])


# ================================================================
# 14. FIXED RESOURCE ALLOCATION
# ================================================================

fixed_cores = int(np.ceil(

    y_train["target_cpu"].max()
    * SAFETY_MARGIN
    / (
        CPU_CAPACITY_PER_CORE
        * CPU_UTILIZATION_TARGET
    )

))

fixed_memory = int(np.ceil(

    y_train["target_memory"].max()
    * SAFETY_MARGIN
    / (
        MEMORY_CAPACITY_PER_GB
        * MEMORY_UTILIZATION_TARGET
    )

))


fixed_cores = int(np.clip(
    fixed_cores,
    1,
    MAX_CPU_CORES
))

fixed_memory = int(np.clip(
    fixed_memory,
    1,
    MAX_MEMORY_GB
))


fixed_cores_array = np.full(
    len(X_test),
    fixed_cores
)

fixed_memory_array = np.full(
    len(X_test),
    fixed_memory
)


fixed_cost = calculate_cost(
    fixed_cores,
    fixed_memory
)

fixed_cost_array = np.full(
    len(X_test),
    fixed_cost
)


# ================================================================
# 15. REACTIVE RESOURCE ALLOCATION
# ================================================================

reactive_allocations = [

    allocate_resources(cpu, memory)

    for cpu, memory in zip(
        actual_cpu,
        actual_memory
    )

]


reactive_cost = np.array([
    x["cost"] for x in reactive_allocations
])


# ================================================================
# 16. COST COMPARISON
# ================================================================

fixed_total_cost = fixed_cost_array.sum()

adaptive_total_cost = adaptive_cost.sum()

reactive_total_cost = reactive_cost.sum()


saving_percentage = (

    (fixed_total_cost - adaptive_total_cost)

    / fixed_total_cost

    * 100

    if fixed_total_cost > 0

    else 0

)


print("\nRESOURCE ALLOCATION RESULTS")

print("-----------------------------------")

print("Fixed CPU cores:", fixed_cores)

print("Fixed memory GB:", fixed_memory)

print(
    "Adaptive average CPU cores:",
    round(adaptive_cores.mean(), 2)
)

print(
    "Adaptive average memory GB:",
    round(adaptive_memory.mean(), 2)
)

print(
    "Fixed estimated cost:",
    round(fixed_total_cost, 3)
)

print(
    "Adaptive estimated cost:",
    round(adaptive_total_cost, 3)
)

print(
    "Reactive estimated cost:",
    round(reactive_total_cost, 3)
)

print(
    "Estimated cost difference (%):",
    round(saving_percentage, 2)
)


# ================================================================
# 17. RESOURCE UTILIZATION AND SLA EVALUATION
# ================================================================

adaptive_cpu_capacity = (
    adaptive_cores * CPU_CAPACITY_PER_CORE
)

adaptive_memory_capacity = (
    adaptive_memory * MEMORY_CAPACITY_PER_GB
)

cpu_utilization = (
    actual_cpu / adaptive_cpu_capacity
) * 100

memory_utilization = (
    actual_memory / adaptive_memory_capacity
) * 100


sla_violations = (

    (actual_cpu > 90)

    | (actual_memory > 92)

    | (actual_cpu > adaptive_cpu_capacity)

    | (actual_memory > adaptive_memory_capacity)

)


print("\nSYSTEM PERFORMANCE")

print(
    "Average CPU utilization:",
    round(cpu_utilization.mean(), 2),
    "%"
)

print(
    "Average memory utilization:",
    round(memory_utilization.mean(), 2),
    "%"
)

print(
    "SLA violation hours:",
    int(sla_violations.sum())
)


# ================================================================
# 18. ACTUAL VS PREDICTED CPU GRAPH
# ================================================================

plt.figure(figsize=(12, 5))

plt.plot(
    actual_cpu[:150],
    label="Actual CPU"
)

plt.plot(
    predicted_cpu[:150],
    label="Predicted CPU"
)

plt.title("Actual vs Predicted CPU Usage")

plt.xlabel("Test Records")

plt.ylabel("CPU Usage (%)")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/cpu_prediction.png"
)

plt.show()


# ================================================================
# 19. ACTUAL VS PREDICTED MEMORY GRAPH
# ================================================================

plt.figure(figsize=(12, 5))

plt.plot(
    actual_memory[:150],
    label="Actual Memory"
)

plt.plot(
    predicted_memory[:150],
    label="Predicted Memory"
)

plt.title("Actual vs Predicted Memory Usage")

plt.xlabel("Test Records")

plt.ylabel("Memory Usage (%)")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/memory_prediction.png"
)

plt.show()


# ================================================================
# 20. ADAPTIVE CPU ALLOCATION GRAPH
# ================================================================

plt.figure(figsize=(12, 5))

plt.step(
    range(len(adaptive_cores)),
    adaptive_cores,
    where="post"
)

plt.title("Adaptive CPU Resource Allocation")

plt.xlabel("Test Records")

plt.ylabel("CPU Cores")

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/adaptive_cpu.png"
)

plt.show()


# ================================================================
# 21. ADAPTIVE MEMORY ALLOCATION GRAPH
# ================================================================

plt.figure(figsize=(12, 5))

plt.step(
    range(len(adaptive_memory)),
    adaptive_memory,
    where="post"
)

plt.title("Adaptive Memory Resource Allocation")

plt.xlabel("Test Records")

plt.ylabel("Memory GB")

plt.grid()

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/adaptive_memory.png"
)

plt.show()


# ================================================================
# 22. CLOUD COST COMPARISON GRAPH
# ================================================================

plt.figure(figsize=(9, 5))

plt.bar(

    ["Fixed", "Adaptive", "Reactive"],

    [
        fixed_total_cost,
        adaptive_total_cost,
        reactive_total_cost
    ]

)

plt.title("Cloud Cost Comparison")

plt.ylabel("Illustrative Total Cost")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/cost_comparison.png"
)

plt.show()


# ================================================================
# 23. MODEL COMPARISON GRAPH
# ================================================================

plt.figure(figsize=(8, 5))

plt.bar(

    ["Random Forest", "Extra Trees"],

    [
        rf_average_mae,
        et_average_mae
    ]

)

plt.title("Machine Learning Model Comparison")

plt.ylabel("Average MAE")

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/model_comparison.png"
)

plt.show()


# ================================================================
# 24. WORKLOAD CLASSIFICATION GRAPH
# ================================================================

class_counts = pd.Series(
    workload_class
).value_counts()

class_order = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]

class_counts = class_counts.reindex(
    class_order,
    fill_value=0
)


plt.figure(figsize=(8, 5))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.title("Cloud Workload Classification")

plt.xlabel("Workload Category")

plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER + "/workload_classification.png"
)

plt.show()


# ================================================================
# 25. FEATURE IMPORTANCE
# ================================================================

if selected_model_name == "Random Forest":

    importance = (
        selected_model.estimators_[0].feature_importances_
    )

    importance_table = pd.DataFrame({

        "Feature": features,

        "Importance": importance

    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")

    print(
        importance_table.head(10).round(4)
    )

    top = importance_table.head(10).sort_values(
        "Importance"
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        top["Feature"],
        top["Importance"]
    )

    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER + "/feature_importance.png"
    )

    plt.show()


# ================================================================
# 26. NEXT-HOUR WORKLOAD PREDICTION
# ================================================================

def predict_next_hour(frame):

    row = {}

    next_timestamp = (
        frame["timestamp"].iloc[-1]
        + pd.Timedelta(hours=1)
    )

    for lag in [1, 2, 3, 6, 12, 24]:

        row[f"cpu_lag_{lag}"] = (
            frame["cpu_usage"].iloc[-lag]
        )

        row[f"memory_lag_{lag}"] = (
            frame["memory_usage"].iloc[-lag]
        )

    row["cpu_rolling_mean"] = (
        frame["cpu_usage"].tail(6).mean()
    )

    row["memory_rolling_mean"] = (
        frame["memory_usage"].tail(6).mean()
    )

    row["cpu_rolling_std"] = (
        frame["cpu_usage"].tail(6).std()
    )

    row["memory_rolling_std"] = (
        frame["memory_usage"].tail(6).std()
    )

    hour = next_timestamp.hour

    day = next_timestamp.dayofweek

    row["hour"] = hour

    row["day"] = day

    row["is_weekend"] = int(day >= 5)

    row["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    row["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    next_X = pd.DataFrame(
        [row],
        columns=features
    )

    forecast = selected_model.predict(
        next_X
    )[0]

    cpu = float(np.clip(forecast[0], 0, 100))

    memory = float(np.clip(forecast[1], 0, 100))

    return cpu, memory, next_timestamp


next_cpu, next_memory, next_time = predict_next_hour(data)


next_workload = classify_workload(
    next_cpu,
    next_memory
)

next_allocation = allocate_resources(
    next_cpu,
    next_memory
)


# ================================================================
# 27. NEXT-HOUR RESOURCE RECOMMENDATION
# ================================================================

print("\n===================================")

print("NEXT HOUR CLOUD RESOURCE PREDICTION")

print("===================================")

print("Forecast timestamp:", next_time)

print(
    "Predicted CPU:",
    round(next_cpu, 2),
    "%"
)

print(
    "Predicted memory:",
    round(next_memory, 2),
    "%"
)

print(
    "Workload category:",
    next_workload
)

print(
    "Recommended CPU cores:",
    next_allocation["cpu_cores"]
)

print(
    "Recommended memory:",
    next_allocation["memory_gb"],
    "GB"
)

print(
    "Estimated hourly cost:",
    round(next_allocation["cost"], 4)
)

print(
    "Allocation status:",
    next_allocation["status"]
)


# ================================================================
# 28. SAVE ALL PROJECT RESULTS
# ================================================================

results = pd.DataFrame({

    "actual_cpu": actual_cpu,

    "predicted_cpu": predicted_cpu,

    "actual_memory": actual_memory,

    "predicted_memory": predicted_memory,

    "workload_class": workload_class,

    "adaptive_cpu_cores": adaptive_cores,

    "adaptive_memory_gb": adaptive_memory,

    "adaptive_cost": adaptive_cost,

    "fixed_cpu_cores": fixed_cores_array,

    "fixed_memory_gb": fixed_memory_array,

    "fixed_cost": fixed_cost_array,

    "reactive_cost": reactive_cost

})

results.to_csv(
    OUTPUT_FOLDER + "/cloud_allocation_results.csv",
    index=False
)


model_comparison.to_csv(
    OUTPUT_FOLDER + "/model_comparison.csv",
    index=False
)


next_recommendation = pd.DataFrame([{

    "forecast_timestamp": next_time,

    "predicted_cpu": next_cpu,

    "predicted_memory": next_memory,

    "workload_class": next_workload,

    "recommended_cpu_cores": next_allocation["cpu_cores"],

    "recommended_memory_gb": next_allocation["memory_gb"],

    "estimated_hourly_cost": next_allocation["cost"],

    "allocation_status": next_allocation["status"]

}])


next_recommendation.to_csv(
    OUTPUT_FOLDER + "/next_hour_recommendation.csv",
    index=False
)


# ================================================================
# 29. FINAL PROJECT SUMMARY
# ================================================================

print("\n")

print("=" * 65)

print("PROJECT EXECUTION COMPLETED")

print("=" * 65)

print("Project Name:", PROJECT_NAME)

print("Machine Learning Model:", selected_model_name)

print("Prediction: CPU and Memory")

print("Anomaly Detection: Isolation Forest")

print("Workload Classification: LOW / MEDIUM / HIGH / CRITICAL")

print("Resource Allocation: Adaptive")

print("Scheduling: Cost-Aware")

print("Evaluation: MAE, RMSE, R2")

print("Fixed Cost:", round(fixed_total_cost, 3))

print("Adaptive Cost:", round(adaptive_total_cost, 3))

print("Estimated Cost Difference:", round(saving_percentage, 2), "%")

print("SLA Violations:", int(sla_violations.sum()))

print("Output Folder:", OUTPUT_FOLDER)

print("\nGENERATED OUTPUT FILES")

for file in sorted(os.listdir(OUTPUT_FOLDER)):

    print(" -", file)

print("\nALL PROJECT OPERATIONS COMPLETED SUCCESSFULLY")

print("=" * 65)