import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ======================================================
# 1. 数据读取
# ======================================================
file_path = r"D:\盘管实验\去除粗细骨料\盘管实验.xlsx"
df = pd.read_excel(file_path)

# 特征与目标变量
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ======================================================
# 2. 数据标准化
# ======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# 3. DBSCAN 聚类分析
# ======================================================
eps = 0.9
min_samples = 9

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(X_scaled)

# 将聚类标签添加为新特征
X["Cluster_Label"] = cluster_labels

# ======================================================
# 4. 划分训练集与测试集
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================================
# 5. 构建并训练模型
# ======================================================
model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

# ======================================================
# 6. 模型预测
# ======================================================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ======================================================
# 7. 模型评估函数
# ======================================================
def evaluate_model(y_true, y_pred):
    """计算并返回R²、MSE、RMSE、MAE、RRMSE等指标"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    rrmse = rmse / np.mean(y_true) * 100
    return r2, mse, rmse, mae, rrmse


# 训练集与测试集评估
r2_train, mse_train, rmse_train, mae_train, rrmse_train = evaluate_model(y_train, y_pred_train)
r2_test, mse_test, rmse_test, mae_test, rrmse_test = evaluate_model(y_test, y_pred_test)

# ======================================================
# 8. 输出评估结果
# ======================================================
print("\n===== 训练集指标 =====")
print(f"R²: {r2_train:.4f}")
print(f"MSE: {mse_train:.4f}")
print(f"RMSE: {rmse_train:.4f}")
print(f"MAE: {mae_train:.4f}")
print(f"RRMSE: {rrmse_train:.2f}%")

print("\n===== 测试集指标 =====")
print(f"R²: {r2_test:.4f}")
print(f"MSE: {mse_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE: {mae_test:.4f}")
print(f"RRMSE: {rrmse_test:.2f}%")

# ======================================================
# 9. 保存预测结果与评估指标
# ======================================================
save_dir = r"D:\盘管实验\测试结果\去除出细骨料\ccc"
os.makedirs(save_dir, exist_ok=True)

# 保存预测结果
train_results = pd.DataFrame({"Actual": y_train, "Predicted": y_pred_train})
test_results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_test})

train_results.to_excel(os.path.join(save_dir, "训练集预测结果.xlsx"), index=False)
test_results.to_excel(os.path.join(save_dir, "测试集预测结果.xlsx"), index=False)

# 保存评估指标
metrics_df = pd.DataFrame({
    "数据集": ["训练集", "测试集"],
    "R²": [r2_train, r2_test],
    "MSE": [mse_train, mse_test],
    "RMSE": [rmse_train, rmse_test],
    "MAE": [mae_train, mae_test],
    "RRMSE(%)": [rrmse_train, rrmse_test]
})
metrics_df.to_excel(os.path.join(save_dir, "模型评估指标.xlsx"), index=False)

print("\n预测结果与评估指标已成功保存。")
print(f"保存路径: {save_dir}")
