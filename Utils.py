import numpy as np
import pandas as pd

from keras.models import Model
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split


def get_data(encoding="OneHot"):
    test = pd.read_csv('./data/UNSW_NB15_testing-set.csv')
    train = pd.read_csv('./data/UNSW_NB15_training-set.csv')

    nTrain = train.shape[0]

    if encoding == "OneHot":
        combined = pd.concat((train, test), axis=0)
        combined = pd.get_dummies(combined.drop(["attack_cat"], axis=1), columns=["proto", "service", "state"])

        train = combined.iloc[:nTrain]
        train.reset_index(inplace=True, drop=True)

        test = combined.iloc[nTrain:]
        test.reset_index(inplace=True, drop=True)

    train.drop(['id'], inplace=True, axis=1)
    test.drop(['id'], inplace=True, axis=1)

    # train.reset_index(inplace=True)
    # test.reset_index(inplace=True)

    return train, test


def get_unsw_data():
    data_df = pd.read_csv('D:\data/bin_data.csv')
    data_df.drop(data_df.columns[0], inplace=True, axis=1)
    train_df, test_df = train_test_split(data_df, test_size=0.4, random_state=0)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df


def get_nsl_data(data_folder="data"):
    selected_list = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                     'dst_bytes', 'wrong_fragment', 'hot', 'logged_in', 'count', 'srv_count',
                     'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                     'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                     'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
    train = pd.read_csv(data_folder + "/KDDTrain.csv")
    test = pd.read_csv(data_folder + "/KDDTest.csv")

    test.label = np.where(test.label == "normal", 0, 1)
    train.label = np.where(train.label == "normal", 0, 1)

    nTrain = train.shape[0]

    train = train[selected_list]
    test = test[selected_list]

    combined = pd.get_dummies(pd.concat((train, test), axis=0), prefix=["protocol_type", "service", "flag"])

    train = combined.iloc[:nTrain]
    test = combined.iloc[nTrain:]

    return train, test


def mse(pred, true):
    result = []
    for sample1, sample2 in zip(pred, true):
        error = sum((sample1.astype("float") - sample2.astype("float")) ** 2)
        error /= float(len(sample2))
        result.append(error)
    return np.array(result)


def get_losses(model, x):
    pred = model.predict(x, verbose=0)
    err = mse(x, pred)
    return err


def get_losses_sub(model, x):
    pred = model.predict(x, verbose=0)
    err = x - pred
    return err


def confidence_intervals(data, confidence=0.97):
    # 置信区间
    n = len(data)
    # mean & standard deviation
    mean, std_dev = np.mean(data), data.std()
    z_critical = stats.norm.ppf(q=confidence)
    margin_of_error = z_critical * (std_dev / np.sqrt(n))
    return [mean - margin_of_error, mean + margin_of_error]


def boxplot(loss):
    total_message = loss
    q1 = np.percentile(total_message, 25)
    q3 = np.percentile(total_message, 75)
    iqr = q3 - q1
    min_value = q1 - 1.5 * iqr
    max_value = q3 + 1.5 * iqr
    low = min_value
    height = max_value

    return [low, height]


def predict_anomaly(model, x, threshold):
    pred = model.predict(x, verbose=0)
    loss = mse(pred, x)
    res = np.where(loss <= threshold, 0, 1)  # anomaly : 1, normal : 0

    return res


def print_metrics(y_eval: np.ndarray, y_pred: np.ndarray, average: str = 'binary'):
    accuracy = metrics.accuracy_score(y_eval, y_pred)
    precision = metrics.precision_score(y_eval, y_pred, average=average)
    recall = metrics.recall_score(y_eval, y_pred, average=average)
    f1 = metrics.f1_score(y_eval, y_pred, average=average)
    metrics_ = [accuracy, precision, recall, f1]
    output = ''.join(str(metr)[:9] + '\t' for metr in metrics_)
    return output


def update_result(clf_pred, total_pred):
    l = len(clf_pred)
    index = 0
    for i in range(0, len(total_pred) - 1):
        if total_pred[i] == 0:
            total_pred[i] = clf_pred[index]
            index += 1
        if index == l:
            break
    return total_pred


def get_encoding_layer(autoencoder):
    encoding_model = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
    return encoding_model
