import os
import warnings

import numpy as np
# np.random.seed(43)
import tensorflow as tf
from imbens.ensemble import SMOTEBoostClassifier
from imbens.ensemble import SelfPacedEnsembleClassifier
from imbens.sampler._under_sampling import InstanceHardnessThreshold
from imbens.sampler.over_sampling import SMOTE
from imbens.sampler.under_sampling import SelfPacedUnderSampler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
# tf.random.set_seed(7)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import Utils

warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", module="pandas")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(tf.test.is_gpu_available())

# np.random.seed(43)

# train_df, test_df = Utils.get_data(encoding='OneHot')
train_df, test_df = Utils.get_nsl_data()
# train_df = pd.read_csv('./data/KDD99_Data/kdd99_train.csv')
# test_df = pd.read_csv('./data/KDD99_Data/kdd99_test.csv')

Scaler = StandardScaler()
train_label = train_df.label
train_df.drop(["label"], axis=1, inplace=True)
X_train, y_train = Scaler.fit_transform(train_df.values), train_label
X_train_ae = X_train[np.where(train_label == 0)]
X_test, y_test = Scaler.transform(test_df.drop(["label"], axis=1)), test_df.label.values

# model = tf.keras.models.load_model('autoencoder_nsl')
# model = tf.keras.models.load_model('autoencoder_kdd')


# train_df_un, test_df_un = Utils.get_unsw_data()
# train_label = train_df_un.label
# train_df_un.drop(["label"], axis=1, inplace=True)
# # X_train, y_train = Scaler.fit_transform(train_df.values), train_label.values
# X_train_un, y_train_un = train_df_un.values, train_label
# y_test_un = test_df_un.label.values
# X_test_un = test_df_un.drop(["label"], axis=1).values


# model = tf.keras.models.load_model('autoencoder')


def only_down_sample():
    clf = SelfPacedEnsembleClassifier()
    # X_en_te = encoder.predict(X_test)
    kf = KFold(n_splits=5)
    for step, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
        batch_train = X_train[tr_index]
        batch_label = y_train[tr_index]
        clf.fit(batch_train, batch_label)
        y_pred = clf.predict(X_test)

        re = Utils.print_metrics(y_test, y_pred)
        print(re)


def isoloation_forest_predict():
    clf = IsolationForest(contamination=0.1)
    # X_en_te = encoder.predict(X_test)
    kf = KFold(n_splits=5)
    for step, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
        batch_train = X_train[tr_index]
        clf.fit(batch_train)
        y_pred = clf.predict(X_test)
        y_pred[y_pred == -1] = 0
        re = Utils.print_metrics(y_test, y_pred)
        print(re)


def elliptic_envelope_predict():
    clf = EllipticEnvelope()
    # X_en_te = encoder.predict(X_test)
    kf = KFold(n_splits=5)
    for step, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
        batch_train = X_train[tr_index]
        clf.fit(batch_train)
        y_pred = clf.predict(X_test)
        y_pred[y_pred == -1] = 0

        re = Utils.print_metrics(y_test, y_pred)
        print(re)


# def smote():
#     sampler = SMOTE()
#     X_total = np.vstack((X_train_un, X_test_un))
#     y_total = np.vstack((y_train_un.values.reshape(-1, 1), y_test_un.reshape(-1, 1))).reshape(-1)
#     X_re, y_re = sampler.fit_resample(X_total, y_total)
#     print(X_re.shape[0])


def smote_compare():
    clf = SMOTEBoostClassifier(estimator=DecisionTreeClassifier())
    batch_train = X_train
    batch_label = y_train
    for i in range(0, 5):
        clf.fit(batch_train, batch_label)
        y_pred = clf.predict(X_test)
        out = Utils.print_metrics(y_test, y_pred)
        print(out)


# def instance_hardness_sampler():
#     inh = InstanceHardnessThreshold()
#     X_total = np.vstack((X_train_un, X_test_un))
#     y_total = np.vstack((y_train_un.values.reshape(-1, 1), y_test_un.reshape(-1, 1))).reshape(-1)
#     X_re, y_re = inh.fit_resample(X_total, y_total)
#     print(X_re.shape[0])


def instance_hardness():
    iht = InstanceHardnessThreshold()
    batch_train = X_train
    batch_label = y_train
    for i in range(0, 5):
        clf = DecisionTreeClassifier()
        X_re, y_re = iht.fit_resample(batch_train, batch_label)
        clf.fit(X_re, y_re)
        y_pred = clf.predict(X_test)
        out = Utils.print_metrics(y_test, y_pred)
        print(out)


def self_paced_check_numbers():
    clf = SelfPacedEnsembleClassifier()
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_train)
    sampler = SelfPacedUnderSampler()
    X_re, y_re = sampler.fit_resample(X_train, y_train, y_pred_proba=y_prob, alpha=0.8,
                                      classes_=np.array([0, 1]), encode_map={0: 0, 1: 1})
    print(X_re.shape[0])


if __name__ == '__main__':
    # only_down_sample()
    # print('--Split--')
    isoloation_forest_predict()
    print('--Split--')
    elliptic_envelope_predict()
    print('--Split--')
    # smote()
    smote_compare()
    print('--Split--')
    instance_hardness()
    # instance_hardness_sampler()
    # self_paced_check_numbers()
