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
from sklearn.multioutput import MultiOutputRegressor

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
        # flag indicating whether to log-transform only XGB
        self._log_transform = False

        # holds Pipelines for XGB + baselines
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
        """Randomized search over XGB parameters, using log1p only if flag set."""
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
        # choose y for tuning
        y_tune = (
            np.log1p(self.y_train)
            if self._log_transform
            else self.y_train
        )
        search.fit(self.X_train, y_tune)
        self.best_params = {
            k.replace("model__", ""): v
            for k, v in search.best_params_.items()
        }
        print("Best XGB hyperparameters:", self.best_params)

    def train_models(self, log_transform: bool = False):
        """Fit XGB (with optional log1p) plus baseline models."""
        if self.X_train is None:
            raise RuntimeError("Call preprocess() first.")

        # store flag
        self._log_transform = log_transform

        # 1) XGBoost
        xgb_model = xgb.XGBRegressor(
            **self.best_params,
            random_state=self.random_state,
            tree_method="hist",
        )
        self.pipelines["XGB"] = Pipeline([("model", xgb_model)])

        # 2) Baselines
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
        self.pipelines["ElasticNet"] = Pipeline(
            [("model", ElasticNet(random_state=self.random_state))]
        )

        # Fit each
        for name, pipe in self.pipelines.items():
            y_fit = (
                np.log1p(self.y_train)
                if (name == "XGB" and self._log_transform)
                else self.y_train
            )
            pipe.fit(self.X_train, y_fit)

    def evaluate_models(self):
        """Print MAE, RMSE, R², SMAPE for all models."""
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
                        denom == 0,
                        0,
                        np.abs(self.y_test - y_pred) / denom,
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
        """Cross-validate on train set, log1p only for XGB if flag set."""
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
            y_cv = (
                np.log1p(self.y_train)
                if (name == "XGB" and self._log_transform)
                else self.y_train
            )
            results = cross_validate(
                pipe,
                self.X_train,
                y_cv,
                cv=kf,
                scoring=scoring,
                n_jobs=-1,
            )
            print(f"\n=== CV Results for {name} ===")
            for metric, scores in results.items():
                if metric.startswith("test_"):
                    val = (
                        -scores.mean()
                        if "neg" in scoring[metric[5:]]
                        else scores.mean()
                    )
                    print(f"{metric}: {val:.4f} ± {scores.std():.4f}")

    def create_plots(self):
        """Residuals, Pred vs Actual, importances & SHAP for XGB only."""
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
        mn, mx = min(y_pred.min(), self.y_test.min()), max(
            y_pred.max(), self.y_test.max()
        )
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
        # set the log-transform flag first
        self._log_transform = log_transform

        # pipeline steps
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_models(log_transform=log_transform)
        self.evaluate_models()
        self.cross_validate(cv_splits=cv_splits)
        if create_plots:
            self.create_plots()


class MultiOutputEmissionsPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        targets: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.df = df
        self.targets = targets
        self.test_size = test_size
        self.random_state = random_state

        # train/test
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        # best XGB params
        self.best_params: dict = {}
        # whether to log-transform all targets
        self._log_transform = False

        # holds pipelines keyed by name
        self.pipelines: dict[str, Pipeline] = {}

    def preprocess(self):
        X = self.df.drop(columns=self.targets)
        y = self.df[self.targets]
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(self, n_iter: int = 30, cv_splits: int = 5):
        """Randomized search on an XGB wrapped in MultiOutputRegressor."""
        param_dist = {
            "estimator__n_estimators": [100, 200, 500, 1000],
            "estimator__learning_rate": [0.005, 0.01, 0.05, 0.1],
            "estimator__max_depth": [3, 5, 7, 9, 11],
            "estimator__subsample": [0.5, 0.7, 1.0],
            "estimator__colsample_bytree": [0.3, 0.5, 0.7, 1.0],
            "estimator__min_child_weight": [1, 2, 5, 10],
            "estimator__gamma": [0, 0.1, 0.3, 0.5],
        }
        base = xgb.XGBRegressor(
            random_state=self.random_state, tree_method="hist"
        )
        mor = MultiOutputRegressor(base)
        pipe = Pipeline([("model", mor)])
        kf = KFold(
            n_splits=cv_splits, shuffle=True, random_state=self.random_state
        )
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions={f"model__{k}": v for k, v in param_dist.items()},
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=kf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )

        # choose y for tuning
        y_tune = np.log1p(self.y_train) if self._log_transform else self.y_train
        search.fit(self.X_train, y_tune)
        # strip off "model__estimator__" prefix
        self.best_params = {
            k.replace("model__estimator__", ""): v
            for k, v in search.best_params_.items()
        }
        print("Best XGB hyperparameters:", self.best_params)

    def train_models(self, log_transform: bool = False):
        """Build & fit all multi‐output pipelines."""
        if self.X_train is None:
            raise RuntimeError("Call preprocess() first.")
        self._log_transform = log_transform

        # 1) XGB
        xgb_base = xgb.XGBRegressor(
            **self.best_params,
            random_state=self.random_state,
            tree_method="hist",
        )
        self.pipelines["XGB"] = Pipeline(
            [("model", MultiOutputRegressor(xgb_base))]
        )

        # 2) Baselines
        self.pipelines["MeanBaseline"] = Pipeline(
            [("model", MultiOutputRegressor(DummyRegressor(strategy="mean")))]
        )
        self.pipelines["MedianBaseline"] = Pipeline(
            [("model", MultiOutputRegressor(DummyRegressor(strategy="median")))]
        )

        # 3) RandomForest supports multi-output natively
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

        # 4) ElasticNet wrapper
        self.pipelines["ElasticNet"] = Pipeline(
            [("model", MultiOutputRegressor(ElasticNet(
                random_state=self.random_state)))]
        )

        # fit
        for name, pipe in self.pipelines.items():
            y_fit = np.log1p(self.y_train) if (name == "XGB" and self._log_transform) else self.y_train
            pipe.fit(self.X_train, y_fit)

    def evaluate_models(self):
        """Print Train & Test MAE, RMSE, R², SMAPE per target."""
        for name, pipe in self.pipelines.items():
            # 1) Test predictions
            y_test_pred = pipe.predict(self.X_test)
            # 2) Train predictions
            y_train_pred = pipe.predict(self.X_train)

            # undo log if needed
            if name == "XGB" and self._log_transform:
                y_test_pred  = np.expm1(y_test_pred)
                y_train_pred = np.expm1(y_train_pred)

            print(f"\n=== {name} ===")
            for i, tgt in enumerate(self.targets):
                # slice out the i-th column
                y_true_test  = self.y_test.values[:, i]
                y_true_train = self.y_train.values[:, i]
                y_pred_test  = y_test_pred[:, i]
                y_pred_train = y_train_pred[:, i]

                # helper to compute metrics
                def compute_metrics(y_true, y_pred):
                    mae  = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2   = r2_score(y_true, y_pred)
                    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
                    smape = np.mean(
                        np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
                    ) * 100
                    return mae, rmse, r2, smape

                tr_mae, tr_rmse, tr_r2, tr_smape = compute_metrics(
                    y_true_train, y_pred_train
                )
                te_mae, te_rmse, te_r2, te_smape = compute_metrics(
                    y_true_test, y_pred_test
                )

                print(
                    f"{tgt:30s}\n"
                    f"  Train → MAE: {tr_mae:.4f}, RMSE: {tr_rmse:.4f}, "
                    f"R²: {tr_r2:.4f}, SMAPE: {tr_smape:.1f}%\n"
                    f"   Test → MAE: {te_mae:.4f}, RMSE: {te_rmse:.4f}, "
                    f"R²: {te_r2:.4f}, SMAPE: {te_smape:.1f}%"
                )

    def cross_validate_per_target(self, cv_splits: int = 5):
        """
        For each model and for each target variable, run CV on X_train → y_train[target]
        and print train/test MAE, RMSE, and R².
        """
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        }
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        for name, pipe in self.pipelines.items():
            print(f"\n=== CV Results for {name} ===")
            model = pipe.named_steps["model"]

            for tgt in self.targets:
                # y must be 2D for MultiOutputRegressor, 1D otherwise
                if isinstance(model, MultiOutputRegressor):
                    y = self.y_train[[tgt]]        # shape (n_samples, 1)
                else:
                    y = self.y_train[tgt]          # shape (n_samples,)

                # apply log1p ONLY if this is the XGB pipeline & flag set
                if name == "XGB" and self._log_transform:
                    y = np.log1p(y)

                results = cross_validate(
                    pipe,
                    self.X_train,
                    y,
                    cv=kf,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                )

                print(f"\n-- Target: {tgt} --")
                for metric in ["MAE", "RMSE", "R2"]:
                    train_scores = results[f"train_{metric}"]
                    test_scores  = results[f"test_{metric}"]

                    # flip sign back for the neg_ metrics
                    if metric in ["MAE", "RMSE"]:
                        train_mean = -train_scores.mean()
                        test_mean  = -test_scores.mean()
                    else:
                        train_mean = train_scores.mean()
                        test_mean  = test_scores.mean()

                    print(
                        f"{metric:5s} | "
                        f"Train: {train_mean:.4f} ± {train_scores.std():.4f}  | "
                        f"Test:  {test_mean:.4f} ± {test_scores.std():.4f}"
                    )

    def plot_feature_importances(self, top_n: int = 10):
            """
            For each target, plot the top_n feature importances from the XGB estimators.
            """
            # grab the multi-output regressor
            mor = self.pipelines["XGB"].named_steps["model"]
            feature_names = self.X_train.columns
            
            n_targets = len(self.targets)
            fig, axes = plt.subplots(n_targets, 1, figsize=(8, 4 * n_targets))
            
            if n_targets == 1:
                axes = [axes]
            
            for i, tgt in enumerate(self.targets):
                # each estimator_[i] is an XGBRegressor
                imp = mor.estimators_[i].feature_importances_
                idx = np.argsort(imp)[-top_n:][::-1]
                
                axes[i].barh(feature_names[idx][::-1], imp[idx][::-1])
                axes[i].set_title(f"Top {top_n} Importances for '{tgt}'")
                axes[i].set_xlabel("Importance")
            
            plt.tight_layout()
            plt.show()

    def plot_residuals(self):
        """
        For each target, scatter residual = true - pred vs predicted.
        Plots are arranged side by side.
        """
        mor = self.pipelines["XGB"].named_steps["model"]
        y_pred = mor.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)
        residuals = self.y_test.values - y_pred

        n = len(self.targets)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, tgt in enumerate(self.targets):
            axes[i].scatter(y_pred[:, i], residuals[:, i], alpha=0.6)
            axes[i].axhline(0, color="k", linestyle="--")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Residual")
            axes[i].set_title(f"Residuals vs Predicted for '{tgt}'")
            axes[i].set_ylim(-40, 40)

        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self):
        """
        For each target, scatter actual vs predicted with a 45° reference line.
        Plots are arranged side by side.
        """
        mor = self.pipelines["XGB"].named_steps["model"]
        y_pred = mor.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)

        n = len(self.targets)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, tgt in enumerate(self.targets):
            y_true = self.y_test.values[:, i]
            mn, mx = min(y_true.min(), y_pred[:, i].min()), max(y_true.max(), y_pred[:, i].max())

            axes[i].scatter(y_true, y_pred[:, i], alpha=0.6)
            axes[i].plot([mn, mx], [mn, mx], "k--", linewidth=1)
            axes[i].set_xlabel("Actual")
            axes[i].set_ylabel("Predicted")
            axes[i].set_title(f"Actual vs Predicted for '{tgt}'")

        plt.tight_layout()
        plt.show()

    def predict(self, df_new: pd.DataFrame, ml_pipeline=None) -> pd.DataFrame:
        """
        Predict all target variables on df_new.
        df_new must contain the same feature columns used for training.
        Returns a DataFrame of shape (n_samples, n_targets).
        """
        X_new = df_new[self.X_train.columns]
        pipeline = ml_pipeline if ml_pipeline is not None else self.pipelines.get("XGB")
        if pipeline is None:
            raise ValueError("No pipeline provided and 'XGB' pipeline not trained.")
        y_pred = pipeline.predict(X_new)
        if self._log_transform:
            y_pred = np.expm1(y_pred)
        return pd.DataFrame(y_pred, columns=self.targets, index=X_new.index)

    
    def run(
        self,
        tune: bool = True,
        log_transform: bool = False,
        cv_splits: int = 5,
        plot_figures: bool = True,
    ):
        self._log_transform = log_transform
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_models(log_transform=log_transform)
        self.evaluate_models()
        self.cross_validate_per_target(cv_splits=cv_splits)
        if plot_figures:
            self.plot_feature_importances()
            self.plot_residuals()
            self.plot_actual_vs_predicted()