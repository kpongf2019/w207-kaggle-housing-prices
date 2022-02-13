import os
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import PowerTransformer, StandardScaler


def read_features(features_loc):
    """
    Combines the created features into dataframes for ML

    :param features_loc: str
        Path of the folder where features are head. Needs to end with `/`
        For example: '../data/modeling/'
    :return X_train, y_train, X_test:
    """
    X_train = [ele for ele in os.listdir(features_loc) if '_train.parquet' in ele]
    X_train = [pd.read_parquet(features_loc + ele) for ele in X_train]
    X_train = pd.concat(X_train, axis=1)

    X_test = [ele for ele in os.listdir(features_loc) if '_test.parquet' in ele]
    X_test = [pd.read_parquet(features_loc + ele) for ele in X_test]
    X_test = pd.concat(X_test, axis=1)

    test_cols_to_drop = [ele for ele in X_test.columns if ele not in X_train.columns]
    X_test = X_test.drop(test_cols_to_drop, axis=1)
    test_cols_to_add = [ele for ele in X_train.columns if ele not in X_test.columns]
    for col in test_cols_to_add:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    y_train = [ele for ele in os.listdir(features_loc) if 'y_var.parquet' == ele]
    y_train = pd.read_parquet(features_loc + y_train[0])
    y_train = y_train['SalePrice']
    return X_train, y_train, X_test


def plot_categorical_univariate_feature(feature, y_var, df_train, df_test=None):
    """

    Creates three plots using the training set, `df_train`:
        * Average value of dependent variable for each feature
        * Scatterplot (stripplot) of dependent variable for each feature
        * Count of each feature

    Calculates and prints number of NaNs for that feature in the traning and testing datasets

    Parameters
    ----------
    feature : str
        Feature to analyze and plot
    y_var : str
        Dependent variable to analyze and plot
    df_train : pd.DataFrame
        Training dataset
    df_test : None or pd.DataFrame
        Testing dataset

    Returns
    -------
    None
    
    Can also be used with iPywidgets Interactions as per below:
    interact(plot_categorical_univariate_feature,
             feature=categorical_feature_list,  # List: categorical features
             y_var=fixed(y_var),  # Str: dependent variable
             df_train=fixed(df_train),  # pd.DataFrame: training set
             df_test=fixed(df_train));  # pd.DataFrame: testing set

    """
    df = df_train.copy()
    print(f"The training set has {df[feature].isna().sum()} nulls.")

    if df_test is not None:
        print(f"The testing set has {df_test[feature].isna().sum()} nulls.")

    df[feature] = df[feature].astype(str).str.strip() + '_'  # Issues with seaborn for float-like
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax = ax.flatten()

    groupby = df[[feature, y_var]].groupby(feature)[y_var]
    avgs = groupby.mean().sort_values(ascending=True)

    sns.pointplot(y=feature, x=y_var, data=df, order=avgs.index, ci='sd', ax=ax[0])
    ax[0].set_title(f"Avg {y_var} with std dev")

    sns.stripplot(y=feature, x=y_var, data=df, order=avgs.index, alpha=0.3, ax=ax[1])
    ax[1].set_title(f"{y_var} vs {feature} data points")
    ax[1].set_ylabel('')
    ax[1].set_yticklabels('')

    sns.countplot(y=feature, data=df, order=avgs.index, ax=ax[2])
    ax[2].set_title(f"Count of {feature}")
    ax[2].set_ylabel('')
    ax[2].set_yticklabels('')

    plt.tight_layout();
    return None


def plot_quantitative_univariate_feature(feature, y_var, df_train, df_test=None):
    """

    Shows average dependent variable for features:
    * Untransformed
    * Log
    * Sqrt
    * Square
    * PowerTransform
    * StandardScale

    Calculates and prints number of NaNs for that feature in the traning and testing datasets

    Parameters
    ----------
    feature : str
        Feature to analyze and plot
    y_var : str
        Dependent variable to analyze and plot
    df_train : pd.DataFrame
        Training dataset
    df_test : None or pd.DataFrame
        Testing dataset

    Returns
    -------
    None

    Can also be used with iPywidgets Interactions as per below:
    interact(plot_quantitative_univariate_feature,
             feature=quantitative_feature_list,  # list: quantitative features
             y_var=fixed(y_var),  # str: dependent variable
             df_train=fixed(df_train),  # pd.DataFrame: training set
             df_test=fixed(df_train));  # pd.DataFrame: testing set

    """
    df = df_train.copy()
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    print(f"The training set has {df[feature].isna().sum()} nulls.")

    if df_test is not None:
        test = df_test.copy()
        test[feature] = pd.to_numeric(test[feature], errors='coerce')
        print(f"The testing set has {test[feature].isna().sum()} nulls.")

    fig, ax = plt.subplots(6, 3, figsize=(12, 10))
    ax = ax.flatten()

    df[f"new_{feature}"] = df[feature]
    sns.scatterplot(x=df[f"new_{feature}"], y=df[y_var], ax=ax[0], alpha=0.2)
    ax[0].set_xlabel(f"{feature}")
    sns.distplot(df[f"new_{feature}"], kde=False, ax=ax[1], bins=30)
    ax[1].set_xlabel(f"{feature} bins")
    ax[1].set_ylabel(f"Hist cnts")
    df[f"new_{feature}"] = pd.cut(df[f"new_{feature}"], bins=30)
    df.groupby([f"new_{feature}"])[y_var].mean().dropna().plot(ax=ax[2], kind='line')
    ax[2].set_xlabel(f"{feature} bins")
    ax[2].set_ylabel(f"{y_var}")

    df[f"new_{feature}"] = np.log(df[feature])
    df[f"new_{feature}"] = df[f"new_{feature}"].replace([np.inf, -np.inf], np.nan)
    sns.scatterplot(x=df[f"new_{feature}"], y=df[y_var], ax=ax[3], alpha=0.2)
    ax[3].set_xlabel(f"Log {feature}")
    sns.distplot(df[f"new_{feature}"], kde=False, ax=ax[4], bins=30)
    ax[4].set_xlabel(f"Log {feature} bins")
    ax[4].set_ylabel(f"Hist cnts")
    df[f"new_{feature}"] = pd.cut(df[f"new_{feature}"], bins=30)
    df.groupby([f"new_{feature}"])[y_var].mean().dropna().plot(ax=ax[5], kind='line')
    ax[5].set_xlabel(f"Log {feature} bins")
    ax[5].set_ylabel(f"{y_var}")

    df[f"new_{feature}"] = np.sqrt(df[feature])
    sns.scatterplot(x=df[f"new_{feature}"], y=df[y_var], ax=ax[6], alpha=0.2)
    ax[6].set_xlabel(f"Sqrt {feature}")
    sns.distplot(df[f"new_{feature}"], kde=False, ax=ax[7], bins=30)
    ax[7].set_xlabel(f"Sqrt {feature} bins")
    ax[7].set_ylabel(f"Hist cnts")
    df[f"new_{feature}"] = pd.cut(df[f"new_{feature}"], bins=30)
    df.groupby([f"new_{feature}"])[y_var].mean().dropna().plot(ax=ax[8], kind='line')
    ax[8].set_xlabel(f"Sqrt {feature} bins")
    ax[8].set_ylabel(f"{y_var}")

    df[f"new_{feature}"] = df[feature] ** 2
    sns.scatterplot(x=df[f"new_{feature}"], y=df[y_var], ax=ax[9], alpha=0.2)
    ax[9].set_xlabel(f"Sq {feature}")
    sns.distplot(df[f"new_{feature}"], kde=False, ax=ax[10], bins=30)
    ax[10].set_xlabel(f"Sq {feature} bins")
    ax[10].set_ylabel(f"Hist cnts")
    df[f"new_{feature}"] = pd.cut(df[f"new_{feature}"], bins=30)
    df.groupby([f"new_{feature}"])[y_var].mean().dropna().plot(ax=ax[11], kind='line')
    ax[11].set_xlabel(f"Sq {feature} bins")
    ax[11].set_ylabel(f"{y_var}")

    pt = PowerTransformer()
    df[f"new_{feature}"] = pt.fit_transform(df[[feature]])
    sns.scatterplot(x=df[f"new_{feature}"], y=df[y_var], ax=ax[12], alpha=0.2)
    ax[12].set_xlabel(f"PT {feature}")
    sns.distplot(df[f"new_{feature}"], kde=False, ax=ax[13], bins=30)
    ax[13].set_xlabel(f"PT {feature} bins")
    ax[13].set_ylabel(f"Hist cnts")
    df[f"new_{feature}"] = pd.cut(df[f"new_{feature}"], bins=30)
    df.groupby([f"new_{feature}"])[y_var].mean().dropna().plot(ax=ax[14], kind='line')
    ax[14].set_xlabel(f"PT {feature} bins")
    ax[14].set_ylabel(f"{y_var}")

    scaler = StandardScaler()
    df[f"new_{feature}"] = scaler.fit_transform(df[[feature]])
    sns.scatterplot(x=df[f"new_{feature}"], y=df[y_var], ax=ax[15], alpha=0.2)
    ax[15].set_xlabel(f"Scaled {feature}")
    sns.distplot(df[f"new_{feature}"], kde=False, ax=ax[16], bins=30)
    ax[16].set_xlabel(f"Scaled {feature} bins")
    ax[16].set_ylabel(f"Hist cnts")
    df[f"new_{feature}"] = pd.cut(df[f"new_{feature}"], bins=30)
    df.groupby([f"new_{feature}"])[y_var].mean().dropna().plot(ax=ax[17], kind='line')
    ax[17].set_xlabel(f"Scaled {feature} bins")
    ax[17].set_ylabel(f"{y_var}")

    plt.tight_layout();
    plt.show()
    return None


def transform_results_to_prices(results):
    results = np.exp(results)
    return results


def predict_transform_from_test_set(estimator, X_test, X_test_index=None):
    results = estimator.predict(X_test)
    results = transform_results_to_prices(results)
    if X_test_index is not None:
        results = pd.DataFrame({'Id': X_test_index, 'SalePrice': results})
    else:
        results = pd.DataFrame({'Id': X_test.index, 'SalePrice': results})
    return results


def plot_residual_prediction_from_train(estimator, X_train, y_train, cv, n_jobs):
    """

    Plots the predicted values vs residuals from cross-validated estimates (each point estimated from model fit
    on other folds. The returned Plotly plot has the index of each point in the hover tips.

    Parameters
    ----------
    estimator : object with fit and predict
    X_train : pd.DataFrame
    y_train : pd.Series
    cv : int
    n_jobs : int

    Returns
    -------
    None

    """
    est = deepcopy(estimator)
    y_train_pred = cross_val_predict(est, X_train.values, y_train.values, cv=cv, n_jobs=n_jobs)
    residual = y_train - y_train_pred
    fig = px.scatter(x=y_train_pred, y=residual, hover_name=y_train.index,
                     labels={'x': f'Predicted y_var', 'y': 'Residual (y_actual - y_pred)'},
                     title='Prediction vs Residual', opacity=0.5)
    fig.show()
    return None


def estimator_predict(data_asarray, estimator, columns):
    """
    Helper for SHAP functions as SHAP Kernel Explainer sends data as a numpy array with no column names
    """
    data_asframe = pd.DataFrame(data_asarray, columns=columns)
    return estimator.predict(data_asframe)


def get_force_plot_for_sample_in_train_loo(estimator, X_train, y_train, index):
    """

    Plots the effect of each feature in pushing the predicted value up or down for a particular index in the
    training set. Note that whichever index is provided will be excluded from the training.

    Parameters
    ----------
    estimator : object with fit and predict
    X_train : pd.DataFrame
    y_train : pd.Series
    index : int
        Index of the data point that you want a SHAP forceplot for

    Returns
    -------
        None

    """
    indices = [index]
    new_X_train = X_train.drop(indices, axis=0)
    new_y_train = y_train.drop(indices, axis=0)
    new_X_val = X_train.loc[indices, :]

    est = deepcopy(estimator)
    est.fit(new_X_train, new_y_train)

    est_pred = partial(estimator_predict, estimator=est, columns=new_X_train.columns)
    explainer = shap.KernelExplainer(est_pred, shap.kmeans(new_X_train.astype(float).values, 100))
    shap_values = explainer.shap_values(new_X_val.astype(float).values, nsamples=500)
    shap_display = shap.force_plot(explainer.expected_value, shap_values[0, :], new_X_val.iloc[0, :])
    display(shap_display)
    return None


def get_feature_importances_for_training(estimator, X_train_df, y_train, cv, tree_shap=False, cv_prediction=True,
                                         max_sample_size=None, plot_feature_importances=True,
                                         return_explainer_if_no_cv=True):
    """

    Performs cross-validation for the training set, shows the feature importances, and returns the SHAP values

    Parameters
    ----------
    estimator : object with fit and predict
    X_train_df : pd.DataFrame
    y_train : pd.DataFrame
    cv : int (only used if cv_prediction==True)
    tree_shap : False (uses KernelExplainer) or True (Uses TreeExplainer - preferred if tree)
    cv_prediction : True or False (fits and explains same data, causing overfitting on the training set)
    max_sample_size : int (max sample size; otherwise performs calculations on a smaller df
    plot_feature_importances : True or False
    return_explainer_if_no_cv : True (returns the Explainer object if cross validation is not used) or False

    Returns
    -------
    shap_values : np.ndarray (array of Shap Values)
    expected_value : float (average expected value)

    """
    if max_sample_size is not None:
        if len(X_train_df) > max_sample_size:
            X_train = X_train_df.sample(max_sample_size)
        else:
            X_train = X_train_df
    else:
        X_train = X_train_df
    if cv_prediction:
        kf = KFold(n_splits=cv)
        shap_values_dfs = []
        explainers = []
        for train_idx_nums, val_idx_nums in kf.split(X_train):
            new_X_train = X_train.iloc[train_idx_nums, :]
            new_y_train = y_train.iloc[train_idx_nums]
            X_val = X_train.iloc[val_idx_nums, :]

            est = deepcopy(estimator)
            est.fit(new_X_train, new_y_train)

            if tree_shap:
                explainer = shap.TreeExplainer(est, model_output='raw')
            else:
                est_pred = partial(estimator_predict, estimator=est, columns=new_X_train.columns)
                explainer = shap.KernelExplainer(est_pred, shap.kmeans(new_X_train.astype(float).values, 100))
            shap_values = explainer.shap_values(X_val.astype(float).values, nsamples=100)
            shap_values_df = pd.DataFrame(shap_values, index=X_val.index)
            shap_values_dfs.append(shap_values_df)
            explainers.append(explainer)
        shap_values_dfs = pd.concat(shap_values_dfs, axis=0, ignore_index=False)
        shap_values_dfs = shap_values_dfs.loc[X_train.index, :]
        shap_values = shap_values_dfs.values
        expected_value = np.mean([explainer.expected_value for explainer in explainers])
    else:
        est = deepcopy(estimator)
        est.fit(X_train, y_train)

        if tree_shap:
            explainer = shap.TreeExplainer(est, model_output='raw')
        else:
            est_pred = partial(estimator_predict, estimator=est, columns=X_train.columns)
            explainer = shap.KernelExplainer(est_pred, shap.kmeans(X_train.astype(float).values, 100))
        shap_values = explainer.shap_values(X_train.astype(float).values, nsamples=100)
        expected_value = explainer.expected_value
    if plot_feature_importances:
        shap_display = shap.summary_plot(shap_values, X_train)
        display(shap_display)
    if return_explainer_if_no_cv and (cv_prediction is False):
        return expected_value, shap_values, explainer
    else:
        return expected_value, shap_values
