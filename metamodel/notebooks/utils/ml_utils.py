import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate


class PreprocessingUtils:

    @staticmethod
    def pca_reduce(df, var_threshold=0.95):
        pca = PCA(n_components=var_threshold)
        Xp = pca.fit_transform(df)
        return pd.DataFrame(Xp,
                            columns=[f"PC{i+1}" for i in range(Xp.shape[1])],
                            index=df.index), pca
    

class RFEmissionsPredictionPipeline:
    def __init__(self, df, target: str = 'total_emissions_last_five_years', test_size: float = 0.2, random_state: int = 42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.df = df
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.pipeline = None
        self.best_params = None

    def preprocess(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(self, n_iter: int = 30, cv_splits: int = 5):
        param_dist = {
            'n_estimators': [200, 500, 1000, 3000],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5]
        }
        rf = RandomForestRegressor(random_state=self.random_state)
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            rf, param_dist, n_iter=n_iter,
            scoring='neg_mean_absolute_error', cv=kf,
            random_state=self.random_state, n_jobs=-1, verbose=1
        )
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train_model(self, log_transform: bool = False):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        model_params = self.best_params or {'n_estimators': 100, 'random_state': self.random_state}
        rf = RandomForestRegressor(**model_params)
        steps = [('scaler', StandardScaler()), ('model', rf)]
        self.pipeline = Pipeline(steps)
        y = np.log1p(self.y_train) if log_transform else self.y_train
        self.pipeline.fit(self.X_train, y)
        self._log_transform = log_transform

    def evaluate_model(self):
        if self.pipeline is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model not trained or data not preprocessed.")
        y_pred = self.pipeline.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)
        print("MAE:", mean_absolute_error(self.y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(self.y_test, y_pred)))
        print("R^2 Score:", r2_score(self.y_test, y_pred))

        # Residual plot
        res = self.y_test - y_pred
        plt.figure(figsize=(8,5))
        plt.scatter(y_pred, res, alpha=0.6)
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        plt.show()

        # Scatter plot
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, self.y_test, alpha=0.6)
        min_val = min(y_pred.min(), self.y_test.min())
        max_val = max(y_pred.max(), self.y_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        plt.xlabel('Predicted Emissions')
        plt.ylabel('Actual Emissions')
        plt.title('Predicted vs. Actual Emissions')
        plt.tight_layout()
        plt.show()

        # SHAP summary plot
        explainer = shap.Explainer(self.pipeline.named_steps['model'], self.X_test)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test, show=True)

    def cross_validate_model(self, cv_splits: int = 5):
        if self.pipeline is None:
            raise ValueError("Model not trained.")
        scoring = {'MAE': 'neg_mean_absolute_error', 'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error'}
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        results = cross_validate(
            self.pipeline, self.X_train, np.log1p(self.y_train) if self._log_transform else self.y_train,
            cv=kf, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        print("\nCross-Validation Results:")
        for metric, scores in results.items():
            if 'test' in metric:
                mean_score = np.mean(-scores) if 'neg' in metric else np.mean(scores)
                print(f"{metric}: Mean={mean_score:.4f}, Std={np.std(scores):.4f}")

    def plot_feature_importances(self, top_n: int = 10):
        if not hasattr(self.pipeline.named_steps['model'], 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_.")
        importances = self.pipeline.named_steps['model'].feature_importances_
        features = self.X_train.columns
        idx = np.argsort(importances)[-top_n:][::-1]
        plt.figure(figsize=(8,5))
        plt.barh(features[idx][::-1], importances[idx][::-1])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.show()

    def run(self, tune: bool = True, log_transform: bool = False):
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform)
        self.evaluate_model()
        self.cross_validate_model()
        self.plot_feature_importances()



class GBEmissionsPredictionPipeline:
    def __init__(self, df: pd.DataFrame, target: str = 'total_emissions_last_five_years', test_size: float = 0.2, random_state: int = 42):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.pipeline = None
        self.best_params = None
        self._log_transform = False

    def preprocess(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    def tune_hyperparameters(self, n_iter: int = 30, cv_splits: int = 5):
        param_dist = {
            'n_estimators': [100, 200, 500, 1000],
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.5, 0.7, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
            'min_child_weight': [1, 2, 5, 10],
            'gamma': [0, 0.1, 0.3, 0.5]
        }
        model = xgb.XGBRegressor(random_state=self.random_state, tree_method='hist')
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_mean_absolute_error',
            cv=kf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(self.X_train, np.log1p(self.y_train))
        self.best_params = search.best_params_
        print("Best hyperparameters:", self.best_params)

    def train_model(self, log_transform: bool = False):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call preprocess() before train_model().")

        model_params = self.best_params or {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': self.random_state}
        model = xgb.XGBRegressor(**model_params)

        steps = [('scaler', StandardScaler())]
        steps.append(('model', model))

        self.pipeline = Pipeline(steps)
        y = np.log1p(self.y_train) if log_transform else self.y_train
        self.pipeline.fit(self.X_train, y)
        self._log_transform = log_transform

    
    def evaluate_baseline(self):
        """
        Evaluate baseline model that predicts the mean of y_train.
        
        Parameters:
            y_true (array-like): Ground truth target values (e.g., test set or full set).
            y_train_mean (float, optional): If provided, use this as the prediction. Otherwise, use y_true.mean().
        
        Returns:
            dict: Dictionary with MAE, RMSE, and R².
        """

        y_train_mean = np.mean(self.y_train)
        
        y_pred_baseline = np.full_like(self.y_test, fill_value=y_train_mean, dtype=np.float64)
        
        mae = mean_absolute_error(self.y_test, y_pred_baseline)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_baseline))
        r2 = r2_score(self.y_test, y_pred_baseline)
        
        print(f"Baseline MAE: {mae:.4f}")
        print(f"Baseline RMSE: {rmse:.4f}")
        print(f"Baseline R²: {r2:.4f}")
        print(f"Predicted Mean: {y_train_mean:.4f}")
        print("-"*50)
    
    def evaluate_model(self):
        if self.pipeline is None:
            raise ValueError("Train the model before evaluation.")

        y_pred = self.pipeline.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2  = r2_score(self.y_test, y_pred)
        # SMAPE calculation
        denominator = (np.abs(self.y_test) + np.abs(y_pred)) / 2
        smape = np.mean(np.where(denominator == 0, 0, np.abs(self.y_test - y_pred) / denominator)) * 100

        print(f"MAE:   {mae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"R²:    {r2:.4f}")
        print(f"SMAPE: {smape:.2f}%")
        print("-"*50)

    def create_plots(self):
        if self.pipeline is None:
            raise ValueError("Train the model before creating plots.")

        y_pred = self.pipeline.predict(self.X_test)
        if self._log_transform:
            y_pred = np.expm1(y_pred)
        residuals = self.y_test - y_pred

        model = self.pipeline.named_steps['model']
        importances = model.feature_importances_
        features = self.X_train.columns
        idx = np.argsort(importances)[-4:][::-1]
        # print("Top 2 feature importances:", features[idx])
        # print("Importances:", importances[idx])


        # First figure: Residuals and Predicted vs Actual
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(0, linestyle='--', color='k')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs. Predicted')
        axes[0].set_ylim(30, -30)  # Set y-limits from 30 to -30

        axes[1].scatter(y_pred, self.y_test, alpha=0.6)
        min_val = min(y_pred.min(), self.y_test.min())
        max_val = max(y_pred.max(), self.y_test.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        axes[1].set_xlabel('Predicted Emissions')
        axes[1].set_ylabel('Actual Emissions')
        axes[1].set_title('Predicted vs. Actual Emissions')

        plt.tight_layout()
        plt.show()

        # Second figure: Feature Importances
        plt.figure(figsize=(8, 5))
        plt.barh(features[idx][::-1], importances[idx][::-1])
        plt.xlabel('Importance')
        plt.title('Top 4 Feature Importances')
        plt.tight_layout()
        plt.show()

        # Third figure: SHAP summary plot (top 4 features)
        explainer = shap.Explainer(model, self.X_test)
        shap_values = explainer(self.X_test)

        top4_features = features[np.argsort(importances)[-4:]].tolist()
        plt.figure(figsize=(10, 7))  # Make the plot larger
        shap.summary_plot(
            shap_values[:, top4_features],
            self.X_test[top4_features],
            show=False
        )
        plt.gca().tick_params(axis='y', labelsize=10)  # Make y-axis (feature) labels smaller
        plt.tight_layout()
        plt.show()

    def cross_validate_model(self, cv_splits: int = 5):
        if self.pipeline is None:
            raise ValueError("Model not trained.")

        scoring = {'MAE': 'neg_mean_absolute_error', 'R2': 'r2', 'RMSE': 'neg_root_mean_squared_error'}
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        results = cross_validate(
            self.pipeline,
            self.X_train,
            np.log1p(self.y_train) if self._log_transform else self.y_train,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        print("\nCross-Validation Results:")
        for metric, scores in results.items():
            if 'test' in metric:
                mean_score = -np.mean(scores) if metric in ['test_MAE', 'test_RMSE'] else np.mean(scores)
                print(f"{metric}: Mean={mean_score:.4f}, Std={np.std(scores):.4f}")

    def run(self, tune: bool = True, log_transform: bool = False, create_plots: bool = True):
        self.preprocess()
        if tune:
            self.tune_hyperparameters()
        self.train_model(log_transform=log_transform)
        self.evaluate_baseline()
        self.evaluate_model()
        self.cross_validate_model()
        if create_plots:
            self.create_plots()