import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from openpyxl.reader.excel import load_workbook
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from selenium import webdriver
from matplotlib import image as mpimg
from math import sqrt

# SHAP初始化
shap.initjs()

# 统一的图形参数
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


# ==================== 初始化模型 ====================
def init_models():
    return {
        'DecisionTree': DecisionTreeRegressor(random_state=31, max_depth=4, min_samples_split=19),
        'RandomForest': RandomForestRegressor(random_state=4832, n_estimators=99, max_depth=11),
        'KNN': KNeighborsRegressor(n_neighbors=11),
        'Lasso': Lasso(alpha=0.000270405, max_iter=1000000),
        'SVR': SVR(kernel='linear', C=1.0, epsilon=0.1),
        'ANN': MLPRegressor(hidden_layer_sizes=(100,), activation='tanh', solver='adam',
                            learning_rate='adaptive', max_iter=1000)
    }


# ==================== 模型训练与评估 ====================
def train_and_evaluate_all(models, X_train, Y_train, X_test):
    Y_train_preds = {}
    Y_test_preds = {}
    for model_name, model in models.items():
        model.fit(X_train, Y_train.ravel())
        Y_train_preds[model_name] = np.round(model.predict(X_train), decimals=2)
        Y_test_preds[model_name] = np.round(model.predict(X_test), decimals=2)
    return Y_train_preds, Y_test_preds


def evaluation(y_test, predictions):
    for model_name, y_pred in predictions.items():
        difference = y_test.ravel() - y_pred
        accuracy = np.sum(np.abs(difference) <= 1.0) / len(y_test) * 100
        print(f"模型: {model_name}")
        print(f"准确度: {accuracy:.2f}%")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, R2: {r2_score(y_test, y_pred):.4f}\n")


# ==================== 结果可视化 ====================
def plot_all_predictions(Y_test, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(Y_test, y_pred, alpha=0.5)
        ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
        ax.set_title(model_name)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
    plt.tight_layout()
    plt.show()


def plot_bland_altman(Y_test, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        mean = (Y_test.ravel() + y_pred.ravel()) / 2
        diff = Y_test.ravel() - y_pred.ravel()
        md, sd = np.mean(diff), np.std(diff)
        ax.scatter(mean, diff, alpha=0.5)
        ax.axhline(md, color='red', linestyle='--')
        ax.axhline(md + 1.96 * sd, color='blue', linestyle='--')
        ax.axhline(md - 1.96 * sd, color='blue', linestyle='--')
        ax.set_title(model_name)
        ax.set_xlabel('Mean')
        ax.set_ylabel('Difference')
    plt.tight_layout()
    plt.show()


# ==================== SHAP解释性分析 ====================
def generate_shap_plots(models, X_train, Y_train):
    shap_values_dict = {}
    for model_name, model in models.items():
        model.fit(X_train, Y_train.ravel())
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        shap_values_dict[model_name] = shap_values.values.mean(axis=0)
        shap.summary_plot(shap_values, X_train, show=False)
    plt.show()
    return shap_values_dict


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 数据加载
    x_train = pd.read_excel('2024数据整合_new.xlsx', sheet_name='x_train')
    y_train = pd.read_excel('2024数据整合_new.xlsx', sheet_name='y_train')
    x_test = pd.read_excel('2024数据整合_new.xlsx', sheet_name='x_test')
    y_test = pd.read_excel('2024数据整合_new.xlsx', sheet_name='y_test')

    x_train_values = x_train.drop(columns=['Original_Index']).values
    y_train_values = y_train.drop(columns=['Original_Index']).values
    x_test_values = x_test.drop(columns=['Original_Index']).values
    y_test_values = y_test.drop(columns=['Original_Index']).values

    # 模型训练与评估
    models = init_models()
    Y_train_preds, Y_test_preds = train_and_evaluate_all(models, x_train_values, y_train_values, x_test_values)

    print("测试集评估结果：")
    evaluation(y_test_values, Y_test_preds)

    # 结果可视化
    plot_all_predictions(y_test_values, Y_test_preds)
    plot_bland_altman(y_test_values, Y_test_preds)

    # SHAP解释性分析
    shap_values_dict = generate_shap_plots(models, x_train_values, y_train_values)
