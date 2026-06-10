import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


def load_and_preprocess_data(filepath="players_22.csv", test_size=0.2, random_state=42):
    print("=" * 60)
    print("Đọc dữ liệu từ  22 Dataset")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)

    target = "value_eur"
    selected_features = [
        "overall", "potential", "age", "wage_eur",
        "height_cm", "weight_kg",
        "international_reputation", "weak_foot", "skill_moves",
        "pace", "shooting", "passing", "dribbling", "defending", "physic",
        "preferred_foot"
    ]

    # Lọc các đặc trưng có tồn tại trong dataset
    selected_features = [col for col in selected_features if col in df.columns]
    print("\nSelected Features:", selected_features)

    data = df[selected_features + [target]].copy()

    print("\nXử lý giá trị thiếu")
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    # Xử lý dữ liệu số (Median)
    numeric_imputer = SimpleImputer(strategy="median")
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # Xử lý dữ liệu phân loại (Most Frequent)
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    print("Missing Values Processed Successfully")

    print("\nMã hóa dữ liệu phân loại")
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    print("Shape After Encoding:", data.shape)

    print("\nXử lý và chia dữ liệu")
    X = data.drop(target, axis=1)
    y = np.log1p(data[target])  # Log transform target

    feature_names = X.columns.tolist()
    X = X.values.astype(np.float32)
    y = y.values.reshape(-1, 1).astype(np.float32)

    print("\n--- TRAIN TEST SPLIT ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")

    print("\nChuẩn hóa đặc trưng ")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "scaler.pkl")
    print("Scaler Saved Successfully!")

    return X_train, X_test, y_train, y_test, feature_names


def evaluate_model(model_name, y_true_log, y_pred_log):
    # Chuyển đổi ngược từ log scale về giá trị gốc của EUR
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    # 1. Tải và tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data("players_22.csv")
    results = {}

    # 2. Huấn luyện Ridge Regression    
    poly = PolynomialFeatures( degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    print("\n[MODEL 1] RIDGE REGRESSION")
    lr_model = Ridge(alpha=10)
    lr_model.fit(X_train_poly, y_train)
    y_pred_lr = lr_model.predict(X_test_poly)
    results["Ridge Regression"] = evaluate_model("Ridge Regression", y_test, y_pred_lr)

    # 3. Huấn luyện Random Forest
    print("\n[MODEL 2] RANDOM FOREST")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train.ravel())
    y_pred_rf = rf_model.predict(X_test).reshape(-1, 1)
    results["Random Forest"] = evaluate_model("Random Forest", y_test, y_pred_rf)

    # 4. Huấn luyện Gradient Boosting
    print("\n[MODEL 3] GRADIENT BOOSTING")
    gb_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
    gb_model.fit(X_train, y_train.ravel())
    y_pred_gb = gb_model.predict(X_test).reshape(-1, 1)
    results["Gradient Boosting"] = evaluate_model("Gradient Boosting", y_test, y_pred_gb)

    # 5. In kết quả so sánh
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<25}{'MAE':>15}{'RMSE':>15}{'R2':>15}")
    print("-" * 80)

    for name, metric in results.items():
        print(
            f"{name:<25}"
            f"{metric['MAE']:>15,.2f}"
            f"{metric['RMSE']:>15,.2f}"
            f"{metric['R2']:>15.4f}"
        )
    print("=" * 80)