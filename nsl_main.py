import os
import warnings

import numpy as np
# np.random.seed(43)
import tensorflow as tf
from imbens.ensemble import SelfPacedEnsembleClassifier
from imbens.sampler import SelfPacedUnderSampler
# tf.random.set_seed(7)
from keras import optimizers, regularizers, backend as K
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import Utils

warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", module="pandas")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(tf.test.is_gpu_available())

np.random.seed(43)

# train_df, test_df = Utils.get_data(encoding='OneHot')
train_df, test_df = Utils.get_nsl_data()
Scaler = StandardScaler()
train_label = train_df.label
train_df.drop(["label"], axis=1, inplace=True)
X_train, y_train = Scaler.fit_transform(train_df.values), train_label.values
X_train_ae = X_train[np.where(train_label == 0)]
X_test, y_test = Scaler.transform(test_df.drop(["label"], axis=1)), test_df.label.values


def AutoencoderKDD(X, latent=12, BS=250, ep=100):
    input_dim = train_df.shape[1]
    latent_space_size = latent
    K.clear_session()
    input_ = Input(shape=(input_dim,))

    layer_1 = Dense(100, activation='tanh')(input_)
    layer_2 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(layer_1)
    layer_3 = Dense(25, activation='tanh')(layer_2)

    encoding = Dense(latent_space_size, activation=None)(layer_3)

    layer_6 = Dense(25, activation='tanh')(encoding)
    layer_7 = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(layer_6)
    layer_8 = Dense(100, activation='tanh')(layer_7)

    decoded = Dense(input_dim, activation=None)(layer_8)

    autoencoder = Model(inputs=input_, outputs=decoded)
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=opt)

    autoencoder.fit(X, X, epochs=ep, validation_split=0.2, batch_size=BS, shuffle=True, verbose=0)

    return autoencoder


def train_save():
    autoencoder = AutoencoderKDD(X_train_ae)
    autoencoder.save('autoencoder_nsl')


def autoencoder_predict():
    kf = KFold(n_splits=5)
    for k, (tr_index, te_index) in enumerate(kf.split(X_train_ae)):
        batch_X = X_train_ae[tr_index]
        autoencoder = AutoencoderKDD(batch_X)
        losses = Utils.get_losses(autoencoder, batch_X)
        thresholds = Utils.confidence_intervals(losses, 0.95)
        threshold = thresholds[1]
        y_pred = Utils.predict_anomaly(autoencoder, X_test, threshold=threshold)
        re = Utils.print_metrics(y_test, y_pred)
        print(re)


def check_resamples():
    print('Origin numbers: ' + str(X_train.shape[0]))
    autoencoder = AutoencoderKDD(X_train_ae)
    losses = Utils.get_losses(autoencoder, X_train_ae)
    thresholds = Utils.boxplot(losses)
    threshold = thresholds[1]
    y_pred_tr = Utils.predict_anomaly(autoencoder, X_train, threshold)
    clf = SelfPacedEnsembleClassifier(n_estimators=363, k_bins=30)
    X_normal_tr = X_train[y_pred_tr == 0]
    y_anomaly_tr = y_train[y_pred_tr == 0]
    clf.fit(X_normal_tr, y_anomaly_tr)
    y_anomaly_prob = clf.predict_proba(X_normal_tr)

    spf = SelfPacedUnderSampler()
    X_re, y_re = spf.fit_resample(X_normal_tr, y_anomaly_tr, y_pred_proba=y_anomaly_prob, alpha=0.8,
                                  classes_=np.array([0, 1]), encode_map={0: 0, 1: 1})
    X_anomaly_tr = X_train[y_pred_tr == 1]
    print('After resample: ', str(X_re.shape[0] + X_anomaly_tr.shape[0]))


def main():
    autoencoder = AutoencoderKDD(X_train_ae)
    losses = Utils.get_losses(autoencoder, X_train_ae)
    thresholds = Utils.boxplot(losses)
    threshold = thresholds[1]
    outs = []
    kf = KFold(n_splits=5, shuffle=False)
    for k, (tr_index, te_index) in enumerate(kf.split(X=X_train, y=y_train)):
        Xk_train = X_train[tr_index]
        yk_train = y_train[tr_index]
        y_pred_tr = Utils.predict_anomaly(autoencoder, Xk_train, threshold)
        X_anomaly_tr = Xk_train[y_pred_tr == 0]
        y_anomaly_tr = yk_train[y_pred_tr == 0]
        clf = SelfPacedEnsembleClassifier(n_estimators=363, k_bins=30)
        clf.fit(X_anomaly_tr, y_anomaly_tr)
        y_pred = Utils.predict_anomaly(autoencoder, X_test, threshold)
        # 提取标记为正常的数据实际应用模型
        X_anomaly = X_test[y_pred == 0]
        y_en_pred = clf.predict(X_anomaly)
        # 集成降采样后将分类结果更新替换
        y_final = Utils.update_result(y_en_pred, y_pred)
        # 计算评价指标
        out = Utils.print_metrics(y_test, y_final)
        outs.append(out)

    for out in outs:
        print(out)


if __name__ == '__main__':
    # train_save()
    # autoencoder_predict()
    main()
    # check_resamples()
