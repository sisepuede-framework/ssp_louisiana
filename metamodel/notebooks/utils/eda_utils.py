import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import zscore

class EDAUtils:

    @staticmethod
    def plot_emissions_histogram(df, column="total_emissions_last_five_years", bins=30, kde=True, title="Distribution of Total Emissions in the Last Five Years"):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], bins=bins, kde=kde)
        plt.title(title)
        plt.xlabel("Total Emissions")
        plt.ylabel("Frequency")
        plt.show()
    
    @staticmethod
    def plot_multiple_histograms(df, columns=None, bins=30, kde=True, title="Histograms of Selected Columns"):
        if columns is None:
            columns = df.columns
        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            sns.histplot(df[col], bins=bins, kde=kde, ax=axes[i])
            axes[i].set_title(f"Histogram of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()

    @staticmethod
    def log_transform_comparison_plot(df, column="total_emissions_last_five_years"):
        y = df[column]
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].hist(y, bins=50)
        ax[0].set_title("Raw values")
        ax[1].hist(np.log1p(y), bins=50)
        ax[1].set_title("Log-transformed values")
        plt.show()

    @staticmethod
    def fit_gmm_clusters(df, column="total_emissions_last_five_years", n_components=2, random_state=0):
        log_y = np.log1p(df[column]).values.reshape(-1, 1)
        gm = GaussianMixture(n_components=n_components, random_state=random_state)
        labels = gm.fit_predict(log_y)

        means = gm.means_.flatten()
        sorted_indices = np.argsort(means)
        label_map = {sorted_indices[0]: 'low', sorted_indices[1]: 'high'}

        df['cluster'] = labels
        df['cluster_name'] = df['cluster'].map(label_map)

        return df

    @staticmethod
    def plot_gmm_log_emissions_scatter(df, column="total_emissions_last_five_years"):
        log_y = np.log1p(df[column])
        plt.figure(figsize=(8,5))
        plt.scatter(df.index, log_y,
                    c=df['cluster'], cmap='coolwarm', alpha=0.6)
        plt.xlabel('Sample index')
        plt.ylabel(f'log1p({column})')
        plt.title('GMM Cluster Assignment on Log-Emissions')
        plt.legend(handles=[
            plt.Line2D([], [], marker='o', color='w', label='low',  markerfacecolor='blue', markersize=8),
            plt.Line2D([], [], marker='o', color='w', label='high', markerfacecolor='red',  markersize=8)
        ])
        plt.show()

    @staticmethod
    def plot_gmm_emissions_histograms(df, column="total_emissions_last_five_years"):
        plt.figure(figsize=(8,5))
        for name, grp in df.groupby('cluster_name'):
            plt.hist(grp[column], bins=50, alpha=0.5, label=name)
        plt.xscale('log')
        plt.xlabel(f'{column} (log scale)')
        plt.ylabel('Count')
        plt.title('Emissions Distribution by GMM Cluster')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_cluster_counts(df):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='cluster_name', palette='Set2', hue='cluster_name')
        plt.title("Count of Samples in Each Cluster")
        plt.xlabel("Cluster Name")
        plt.ylabel("Count")
        plt.show()

    @staticmethod
    def plot_all_histograms(df, bins=30, max_cols=4, figsize_per_plot=(4,3)):
        n_features = len(df.columns)
        n_rows = int(np.ceil(n_features / max_cols))
        fig, axes = plt.subplots(n_rows, max_cols,
                                 figsize=(figsize_per_plot[0]*max_cols, figsize_per_plot[1]*n_rows))
        axes = axes.flatten()
        for ax, col in zip(axes, df.columns):
            ax.hist(df[col].dropna(), bins=bins, edgecolor='black')
            ax.set_title(col)
        for ax in axes[n_features:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def find_constant_columns(df):
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df)
        constant_mask = ~selector.get_support()
        return list(df.columns[constant_mask])

    @staticmethod
    def find_near_constant_columns(df, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        near_constant_mask = ~selector.get_support()
        return list(df.columns[near_constant_mask])

    @staticmethod
    def numeric_summary(df):
        desc = df.describe().T
        desc['skew'] = df.skew()
        desc['kurtosis'] = df.kurtosis()
        desc['missing_pct'] = df.isna().mean() * 100
        return desc[['mean','std','skew','kurtosis', 'min','25%','50%','75%','max','missing_pct']]

    @staticmethod
    def find_outlier_columns(df, z_thresh=3.0):
        zs = df.apply(zscore).abs()
        outlier_pct = (zs > z_thresh).sum() / len(df) * 100
        return outlier_pct[outlier_pct>0].to_dict()
    
    @staticmethod
    def get_target_var_corr(df, target_var, threshold=0.1, return_dict=False):
        corr = df.corr()[target_var].abs()
        corr_dict = corr[corr > threshold].sort_values(ascending=False).to_dict()
        print("\n" + "="*50)
        print(f"Correlation with target variable '{target_var}' (threshold: {threshold}):")
        print("-"*50)
        for col, value in corr_dict.items():
            print(f"{col:<30} : {value:.4f}")
        print("="*50 + "\n")

        if return_dict:
            return corr_dict
        else:
            return None
    @staticmethod
    def check_for_multicollinearity(df, threshold=0.8):
        corr_matrix = df.corr().abs()
        to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] >= threshold:
                    colname = corr_matrix.columns[i]
                    to_drop.add(colname)
        print(f"Columns to drop due to multicollinearity (threshold={threshold}): {to_drop}")
        return list(to_drop)


class DataCleaningUtils:

    @staticmethod
    def remove_outliers(df, z_thresh=3.0):
        zs = df.apply(zscore).abs()
        outlier_mask = (zs > z_thresh).any(axis=1)
        return df[~outlier_mask]

    