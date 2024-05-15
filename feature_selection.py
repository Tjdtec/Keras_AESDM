import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

train = pd.read_csv('./data/KDDTrain.csv')
test = pd.read_csv('./data/KDDTest.csv')

test.label = np.where(test.label == "normal", 0, 1)
train.label = np.where(train.label == "normal", 0, 1)

model = XGBClassifier()

# 创建 LabelEncoder 对象
label_encoder = LabelEncoder()

# 对指定列进行编码转换
columns_to_encode = ["protocol_type", "service", "flag"]
for column in columns_to_encode:
    train[column] = label_encoder.fit_transform(train[column])

X = train.iloc[:, :-1]
y = train.iloc[:, -1]
# 使用 SelectFromModel 进行特征选择
selector = SelectFromModel(model)

selector.fit(X, y)

# 获取特征的重要性得分
feature_importances = selector.estimator_.feature_importances_

# 设置阈值，选择至少 20 个重要特征
threshold = sorted(feature_importances, reverse=True)[20]  # 取第 20 大的重要性得分作为阈值

# 获取被选择的特征的索引
selected_indices = [i for i, importance in enumerate(feature_importances) if importance >= threshold]

# 获取被选择的特征的列名
selected_columns = X.columns[selected_indices]

# 打印被选择的特征的列名
print("Selected columns:", selected_columns)
