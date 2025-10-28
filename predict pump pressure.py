import pandas as pd
import numpy as np
import joblib
import os
from sklearn.neighbors import NearestNeighbors

# 路径设置
model_path = r"D:\盘管实验\模型\pump_pressure_model.pkl"
scaler_path = r"D:\盘管实验\模型\scaler.pkl"
train_features_path = r"D:\盘管实验\模型\X_train_scaled.npy"
cluster_labels_path = r"D:\盘管实验\模型\dbscan_labels.npy"
data_path = r"The user's test data.xlsx"
output_path = r"Save the result"

# 加载模型和标准化器
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 加载训练集标准化特征和聚类标签
X_train_scaled = np.load(train_features_path)
train_cluster_labels = np.load(cluster_labels_path)

# 用 NearestNeighbors 建立索引
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train_scaled)

# 读取待预测数据
data = pd.read_excel(data_path)
if 'pressure' in data.columns:
    X = data.drop(columns=['pressure'])
    y_true = data['pressure']
else:
    X = data.copy()
    y_true = None

# 标准化
X_scaled = scaler.transform(X)

# 最近邻映射得到 Cluster_Label
distances, indices = nn.kneighbors(X_scaled)
cluster_labels = train_cluster_labels[indices.flatten()]
X["Cluster_Label"] = cluster_labels

# 对齐特征列
expected_features = model.feature_names_in_
for col in expected_features:
    if col not in X.columns:
        X[col] = 0
X = X[expected_features]

# 预测
y_pred = model.predict(X)

# 保存结果
result = data.copy()
result["Cluster_Label"] = cluster_labels
result["Predicted_pressure"] = y_pred

os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.to_excel(output_path, index=False)
