import warnings

import numpy as np
import pandas as pd
# np.random.seed(43)
import tensorflow as tf
from imbens.ensemble import SelfPacedEnsembleClassifier
# tf.random.set_seed(7)
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
import Utils

train_df = pd.read_csv('./data/KDD99_Data/kdd99_train.csv')
test_df = pd.read_csv('./data/KDD99_Data/kdd99_test.csv')

Scaler = StandardScaler()
train_label = train_df.label
train_df.drop(["label"], axis=1, inplace=True)
X_train, y_train = Scaler.fit_transform(train_df.values), train_label.values
X_train_ae = X_train[np.where(train_label == 0)]
X_test, y_test = Scaler.transform(test_df.drop(["label"], axis=1)), test_df.label.values


def fit_kdd_AE(X):
    input_dim = X.shape[1]
    latent_space_size = 12
    K.clear_session()
    input_ = Input(shape=(input_dim,))

    layer_1 = Dense(100, activation="tanh")(input_)
    layer_2 = Dense(50, activation="tanh")(layer_1)
    layer_3 = Dense(25, activation="tanh")(layer_2)

    encoding = Dense(latent_space_size, activation=None)(layer_3)

    layer_5 = Dense(25, activation="tanh")(encoding)
    layer_6 = Dense(50, activation="tanh")(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim, activation=None)(layer_7)

    autoencoder = Model(inputs=input_, outputs=decoded)
    # opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer="Adam")
    # autoencoder.summary()
    # Fit autoencoder
    autoencoder.fit(X, X, epochs=20, validation_split=0.2, batch_size=100, shuffle=True, verbose=0)

    return autoencoder


def train_save():
    autoencoder = fit_kdd_AE(X_train_ae)
    autoencoder.save('autoencoder_kdd')


def autoencoder_predict():
    kf = KFold(n_splits=5)
    for k, (tr_index, te_index) in enumerate(kf.split(X_train_ae)):
        batch_X = X_train_ae[tr_index]
        autoencoder = fit_kdd_AE(batch_X)
        losses = Utils.get_losses(autoencoder, batch_X)
        thresholds = Utils.confidence_intervals(losses, 0.95)
        threshold = thresholds[1]
        y_pred = Utils.predict_anomaly(autoencoder, X_test, threshold=threshold)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)
        re = Utils.result_concat(acc=accuracy, recall=recall, precision=precision, f1_score=f1)
        print(re)


def main():
    autoencoder = tf.keras.models.load_model('autoencoder_kdd')
    losses = Utils.get_losses(autoencoder, X_train_ae)
    thresholds = Utils.confidence_intervals(losses, 0.95)
    threshold = thresholds[1]
    kf = KFold(n_splits=5, shuffle=False)
    for k, (tr_index, te_index) in enumerate(kf.split(X=X_train, y=y_train)):
        # 计算分类阈值
        Xk_train = X_train[tr_index]
        yk_train = y_train[tr_index]
        clf = SelfPacedEnsembleClassifier(n_estimators=363, k_bins=30)
        clf.fit(Xk_train, yk_train)

        y_pred = Utils.predict_anomaly(autoencoder, X_test, threshold=threshold)
        # 提取标记为正常的数据实际应用模型
        X_anomaly = X_test[y_pred == 0]
        # 获取数据的编码层分布
        y_en_pred = clf.predict(X_anomaly)

        # 集成降采样后将分类结果更新替换
        y_final = Utils.update_result(y_en_pred, y_pred)
        # 计算评价指标
        accuracy = accuracy_score(y_true=y_test, y_pred=y_final)
        precision = precision_score(y_true=y_test, y_pred=y_final)
        recall = recall_score(y_true=y_test, y_pred=y_final)
        f1 = f1_score(y_true=y_test, y_pred=y_final)
        re = Utils.result_concat(acc=accuracy, recall=recall, precision=precision, f1_score=f1)
        print(re)


if __name__ == '__main__':
    # train_save()
    autoencoder_predict()
    # main()
