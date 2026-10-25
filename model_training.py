import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("amazon_cleaned.csv")

numeric_features = [
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating_count"
]

categorical_features = [
    "main_category",
    "brand"
]

features = numeric_features + categorical_features
target = "rating"

X = df[features]
y = df[target]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

comparison_results = []

for name, candidate in models.items():
    if name == "Baseline":
        candidate.fit(y_train.to_frame(), y_train)
        predictions = candidate.predict(y_test.to_frame())
    else:
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    comparison_results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "R2 Score": round(r2, 3)
    })

comparison = pd.DataFrame(comparison_results).sort_values(
    by="MAE"
)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

best_name = comparison.iloc[0]["Model"]
best_model = models[best_name]

print("\nBest model:", best_name)

if best_name == "Baseline":
    print("Baseline performed best. Improve the features before deployment.")
else:
    joblib.dump(best_model, "amazon_rating_model.pkl")
    print("Best model saved successfully!")

    predictions = best_model.predict(X_test)

    prediction_results = pd.DataFrame({
        "Actual Rating": y_test.values,
        "Predicted Rating": predictions.round(2)
    })

    prediction_results["Error"] = (
        prediction_results["Actual Rating"]
        - prediction_results["Predicted Rating"]
    )

    prediction_results["Absolute Error"] = (
        prediction_results["Error"].abs()
    )

    prediction_results.to_csv(
        "rating_predictions.csv",
        index=False
    )

    print("\nFirst 10 Predictions:")
    print(prediction_results.head(10).to_string(index=False))

    print("\nLargest Prediction Errors:")
    print(
        prediction_results
        .sort_values("Absolute Error", ascending=False)
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\nMean Actual Rating:", round(y_test.mean(), 3))
    print("Mean Predicted Rating:", round(predictions.mean(), 3))

    if hasattr(best_model.named_steps["regressor"], "feature_importances_"):
        feature_names = (
            best_model.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        importances = (
            best_model.named_steps["regressor"]
            .feature_importances_
        )

        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(
            by="Importance",
            ascending=False
        )

        print("\nTop 15 Important Features:")
        print(feature_importance.head(15).to_string(index=False))

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs Predicted Amazon Product Ratings")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.show()