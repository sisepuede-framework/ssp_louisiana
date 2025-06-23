import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EmissionsPredictionPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        self.best_params: dict = {}
        self._log_transform = False
        self.pipelines: dict[str, Pipeline] = {}

    def preprocess(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(self, n_iter: int = 30, cv_splits: int = 5):
        param_dist = {
            "n_estimators": [100, 200, 500, 1000],
            "learning_rate": [0.005, 0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7, 9, 11],
            "subsample": [0.5, 0.7, 1.0],
            "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
            "min_child_weight": [1, 2, 5, 10],
            "gamma": [0, 0.1, 0.3, 0.5],
        }
        xgb_pipe = Pipeline(
            [
                (
                    "model",
                    xgb.XGBRegressor(
                        random_state=self.random_state, tree_method="hist"
                    ),
                )
            ]
        )
        kf = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            estimator=xgb_pipe,
            param_distributions={f"model__{k}": v for k, v in param_dist.items()},
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=kf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = {
            k.replace("model__", ""): v
            for k, v in search.best_params_.items()
        }
        print("Best XGB hyperparameters:", self.best_params)

    def train_models(self, log_transform: bool = False):
        if self.X_train is None:
            raise RuntimeError("Call preprocess() first.")
        self._log_transform = log_transform

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            **self.best_params,
            random_state=self.random_state,
            tree_method="hist",
        )
        self.pipelines["XGB"] = Pipeline([("model", xgb_model)])

        # Baselines
        self.pipelines["MeanBaseline"] = Pipeline(
            [("model", DummyRegressor(strategy="mean"))]
        )
        self.pipelines["MedianBaseline"] = Pipeline(
            [("model", DummyRegressor(strategy="median"))]
        )
        self.pipelines["RandomForest"] = Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        random_state=self.random_state, n_jobs=-1
                    ),
                )
            ]
        )
        # **Elastic Net baseline**
        self.pipelines["ElasticNet"] = Pipeline(
            [("model", ElasticNet(random_state=self.random_state))]
        )

        # Fit all pipelines
        for name, pipe in self.pipelines.items():
            y_train = (
                np.log1p(self.y_train)
                if (name == "XGB" and log_transform)
                else self.y_train
            )
            pipe.fit(self.X_train, y_train)

    def evaluate_models(self):
        for name, pipe in self.pipelines.items():
            y_pred = pipe.predict(self.X_test)
            if name == "XGB" and self._log_transform:
                y_pred = np.expm1(y_pred)

            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            denom = (np.abs(self.y_test) + np.abs(y_pred)) / 2
            smape = (
                np.mean(
                    np.where(
                        denom == 0, 0, np.abs(self.y_test - y_pred) / denom
                    )
                )
                * 100
            )
            print(
                f"{name:15s} → MAE: {mae:.4f}, "
                f"RMSE: {rmse:.4f}, R²: {r2:.4f}, SMAPE: {smape:.1f}%"
            )

    def cross_validate(
        self, cv_splits: int = 5, model_names: list[str] | None = None
    ):
        if model_names is None:
            model_names = list(self.pipelines.keys())
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "R2": "r2",
            "RMSE": "neg_root_mean_squared_error",
        }
        kf = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for name in model_names:
            pipe = self.pipelines[name]
            y_train = (
                np.log1p(self.y_train)
                if (name == "XGB" and self._log_transform)
                else self.y_train
            )
            results = cross_validate(
                pipe, self.X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1
            )
            print(f"\n=== CV Results for {name} ===")
            for metric, scores in results.items():
                if metric.startswith("test_"):
                    val = -scores.mean() if "neg" in scoring[metric[5:]] else scores.mean()
                    print(f"{metric}: {val:.4f} ± {scores.std():.4f}")

    def create_plots(self):
        if "XGB" not in self.pipelines:
            raise RuntimeError("Train XGB first.")
        y_pred = self.pipelines["XGB"].predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)
        residuals = self.y_test - y_pred

        model = self.pipelines["XGB"].named_steps["model"]
        importances = model.feature_importances_
        features = self.X_train.columns
        idx = np.argsort(importances)[-4:][::-1]

        # Residuals & Pred vs Actual
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(0, linestyle="--", color="k")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs. Predicted")
        axes[1].scatter(y_pred, self.y_test, alpha=0.6)
        mn, mx = min(y_pred.min(), self.y_test.min()), max(y_pred.max(), self.y_test.max())
        axes[1].plot([mn, mx], [mn, mx], "k--")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")
        axes[1].set_title("Pred vs. Actual")
        plt.tight_layout()
        plt.show()

        # Feature importances
        plt.figure(figsize=(8, 5))
        plt.barh(features[idx][::-1], importances[idx][::-1])
        plt.title("Top 4 Feature Importances")
        plt.tight_layout()
        plt.show()

        # SHAP summary
        explainer = shap.Explainer(model, self.X_test)
        shap_vals = explainer(self.X_test)
        top4 = features[np.argsort(importances)[-4:]].tolist()
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals[:, top4], self.X_test[top4], show=False)
        plt.tight_layout()
        plt.show()

    def run(
        self,
        tune: bool = True,
        log_transform: bool = False,
        cv_splits: int = 5,
        create_plots: bool = True,
    ):
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_models(log_transform=log_transform)
        self.evaluate_models()
        self.cross_validate(cv_splits=cv_splits)
        if create_plots:
            self.create_plots()
