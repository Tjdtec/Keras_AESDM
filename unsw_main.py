import os
import warnings

import numpy as np
# np.random.seed(43)
import tensorflow as tf
# tf.random.set_seed(7)
from imbens.ensemble import SelfPacedEnsembleClassifier
from imbens.sampler import SelfPacedUnderSampler
from keras import regularizers, backend as K
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import KFold

import Utils

warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", module="pandas")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(tf.test.is_gpu_available())

np.random.seed(43)

# train_df, test_df = Utils.get_data(encoding='OneHot')
# train_df, test_df = Utils.get_nsl_data()

train_df, test_df = Utils.get_unsw_data()
# Scaler = StandardScaler()
# Scaler = MinMaxScaler()
train_label = train_df.label
train_df.drop(["label"], axis=1, inplace=True)
# X_train, y_train = Scaler.fit_transform(train_df.values), train_label.values
X_train, y_train = train_df.values, train_label
X_train_ae = X_train[np.where(train_label == 0)]
y_test = test_df.label.values
X_test = test_df.drop(["label"], axis=1).values


def AutoEncoderBase(X, lr=0.001, l2=0.001, epoch=100, batch_size=100):
    input_dim = X.shape[1]
    latent_space_size = 15
    K.clear_session()
    input_ = Input(shape=(input_dim,))

    layer_1 = Dense(100, activation='tanh')(input_)
    layer_2 = Dense(70, activation='tanh', kernel_regularizer=regularizers.l2(l2))(layer_1)
    layer_3 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(l2))(layer_2)

    encoding = Dense(latent_space_size, activation=None, kernel_regularizer=regularizers.l2(0.01))(layer_3)

    layer_5 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(l2))(encoding)
    layer_6 = Dense(70, activation='tanh', kernel_regularizer=regularizers.l2(l2))(layer_5)
    layer_7 = Dense(100, activation='tanh')(layer_6)

    decoded = Dense(input_dim, activation=None)(layer_7)

    autoencoder = Model(inputs=input_, outputs=decoded)
    autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer="adam")
    autoencoder.fit(X, X, epochs=epoch, validation_split=0.2, batch_size=batch_size, shuffle=True, verbose=1)

    return autoencoder


def train_save():
    autoencoder = AutoEncoderBase(X_train_ae)
    autoencoder.save('autoencoder')


def autoencoder_predict():
    kf = KFold(n_splits=5)
    outs = []
    for k, (tr_index, te_index) in enumerate(kf.split(X_train_ae)):
        batch_X = X_train_ae[tr_index]
        autoencoder = AutoEncoderBase(batch_X)
        # autoencoder.save('./test/autoencoder')
        # autoencoder = tf.keras.models.load_model('autoencoder')
        losses = Utils.get_losses(autoencoder, batch_X)
        # losses = Utils.get_losses_sub(autoencoder, batch_X)
        # thresholds = Utils.confidence_intervals(losses, 0.95)
        thresholds = Utils.boxplot(losses)
        threshold = np.abs(thresholds[0])
        y_pred = Utils.predict_anomaly(autoencoder, X_test, threshold=threshold)
        out = Utils.print_metrics(y_test, y_pred)
        outs.append(out)
    for out in outs:
        print(out)


def check_resamples():
    print('Origin numbers: ' + str(X_train.shape[0]+X_test.shape[0]))
    # autoencoder = AutoEncoderBase(X_train_ae)
    autoencoder = tf.keras.models.load_model('autoencoder')
    losses = Utils.get_losses(autoencoder, X_train_ae)
    thresholds = Utils.boxplot(losses)
    threshold = thresholds[1]
    y_pred_tr = Utils.predict_anomaly(autoencoder, X_train, threshold)
    clf = SelfPacedEnsembleClassifier(n_estimators=363, k_bins=30)
    X_normal_tr = X_train[y_pred_tr == 0]
    y_anomaly_tr = y_train[y_pred_tr == 0]
    clf.fit(X_normal_tr, y_anomaly_tr)

    X_total = np.vstack((X_train, X_test))
    y_total = np.vstack((y_train.values.reshape(-1, 1), y_test.reshape(-1, 1))).reshape(-1)
    y_pred = Utils.predict_anomaly(autoencoder, X_total, threshold)

    X_test_majority = X_total[y_pred == 0]
    y_test_majority = y_total[y_pred == 0]
    spf = SelfPacedUnderSampler()
    y_anomaly_prob = clf.predict_proba(X_test_majority)
    X_re, y_re = spf.fit_resample(X_test_majority, y_test_majority, y_pred_proba=y_anomaly_prob, alpha=0.8,
                                  classes_=np.array([0, 1]), encode_map={0: 0, 1: 1})
    X_anomaly_tr = X_train[y_pred_tr == 1]
    print('After resample: ', str(X_re.shape[0] + X_anomaly_tr.shape[0]))


def main():
    kf = KFold(n_splits=5, shuffle=False)
    outs = []
    autoencoder = AutoEncoderBase(X_train_ae)
    # 计算分类阈值
    # losses = Utils.get_losses(autoencoder, X_train_ae)
    # thresholds = Utils.confidence_intervals(losses, 0.95)
    # threshold = thresholds[1]
    losses = Utils.get_losses(autoencoder, X_train_ae)
    thresholds = Utils.boxplot(losses)
    threshold = thresholds[1]

    for k, (tr_index, te_index) in enumerate(kf.split(X=X_train, y=y_train)):
        Xk_train = X_train[tr_index]
        yk_train = y_train[tr_index]
        y_pred_tr = Utils.predict_anomaly(autoencoder, Xk_train, threshold=threshold)
        # 提取标记为正常的数据实际应用模型
        X_anomaly_tr = Xk_train[y_pred_tr == 0]
        y_anomaly_tr = yk_train[y_pred_tr == 0]
        clf = SelfPacedEnsembleClassifier(n_estimators=363, k_bins=33)
        clf.fit(X_anomaly_tr, y_anomaly_tr)
        # Testing
        y_pred = Utils.predict_anomaly(autoencoder, X_test, threshold=threshold)
        X_anomaly = X_test[y_pred == 0]
        y_en_pred = clf.predict(X_anomaly)
        # 集成降采样后将分类结果更新替换
        y_final = Utils.update_result(y_en_pred, y_pred)
        # 计算评价指标
        out = Utils.print_metrics(y_eval=y_test, y_pred=y_final)
        outs.append(out)
    for out in outs:
        print(out)


if __name__ == '__main__':
    # train_save()
    # autoencoder_predict()
    # main()
    check_resamples()
