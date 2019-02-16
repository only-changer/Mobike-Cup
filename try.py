from __future__ import division
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, Conv2D
from keras.optimizers import Adam
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid
from const import ModelChoice, CITY_BLOCK_DICT, FeatureChoice, ReducerChoice, PATH_PATTERN, TARGET, FEATURE_DICT
from const import LOG_DIR, ScaleChoice
import pywt
from sklearn.metrics import mean_squared_error
import pandas


def entropy_evaluation(model_name, y_test, y_predict, label='Test', baseline_flag=False):
    y_predict = y_predict.flatten()

    y_predict_ratio = y_predict / np.sum(y_predict)
    y_test_ratio = y_test / np.sum(y_test)

    y_test_ratio = np.clip(y_test_ratio, 1e-08, 1)
    y_predict_ratio = np.clip(y_predict_ratio, 1e-08, 1)

    # cross_entropy = np.sum(y_test_ratio * (np.log(1 / y_predict_ratio)))
    kl_divergence = np.sum(y_test_ratio * (np.log(y_test_ratio / (y_predict_ratio))))
    rmlse = np.sqrt(mean_squared_error(np.log(y_test_ratio), np.log(y_predict_ratio)))
    print('========================================================================')
    print('Model %s Performance:' % (model_name,))
    # print label, 'Cross Entropy: ', cross_entropy
    if baseline_flag:
        y_base_ratio = np.ones_like(y_test) / len(y_test)
        print('Baseline KL Divergence: ', np.sum(y_test_ratio * (np.log(y_test_ratio / y_base_ratio))))
        print('Baseline RMLSE: ', np.sqrt(mean_squared_error(np.log(y_test_ratio), np.log(y_base_ratio))))

    print(label, 'KL Divergence: ', kl_divergence)
    print(label, 'RMLSE: ', rmlse)
    return np.sum(y_test), kl_divergence, rmlse


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, use_bias=True):
    x = tf.layers.dense(x, units=hidden_sizes[0], activation=activation, use_bias=use_bias,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    for h in hidden_sizes[1:-1]:
        x = tf.layers.dense(x, units=h, activation=activation,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, use_bias=use_bias,
                           kernel_initializer=tf.contrib.layers.xavier_initializer())


def dwt_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    print(tf.rank(y_true))
    print(tf.rank(y_pred))
    # raw_input("\n")
    return K.mean((K.square(abs(y_pred - y_true) * 100)), axis=-1)


fixed_params = [42, 40, 27, 15, 49, 15, 32, 29, 35, 20, 37, 21, 23, 12, 9, 2, 38, 3, 3, 3, 3, 22, 2, 4, 2, 4, 34, 36, 8]
best = 1000
bestn = 0
my_ans = []
for e in range(45):
    my_ans.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
my_ans = np.array(my_ans)

for gogogo in range(29):

    each_max = 10
    each_ans = []
    qaq = 0
    for q in range(1):
        qaq = fixed_params[gogogo]
        train_dfs = [pd.read_csv(PATH_PATTERN)]
        train_df = pd.concat(train_dfs)

        test_dfs = [pd.read_csv(PATH_PATTERN)]
        test_df = pd.concat(test_dfs)

        x_train = train_df[FEATURE_DICT[FeatureChoice.all]].values
        y_train = train_df[TARGET].values

        x_test = test_df[FEATURE_DICT[FeatureChoice.all]].values
        y_test = test_df[TARGET].values

        check = test_df['loss'].values
        reducer = FactorAnalysis(n_components=qaq)
        estimator = [('scaler', StandardScaler()), ('reducer', reducer)]

        pipe = Pipeline(estimator)
        x = pipe.fit_transform(np.vstack([x_train, x_test]))
        split_index = len(x_train)
        x_train = x[:split_index, :]
        x_test = x[split_index:, :]

        print(np.shape(x_train))

        X = []
        Y = []
        x_test = []
        y_test = []
        cnt = 0
        for k in range(len(x_train)):
            i = k % 61
            j = k // 61
            if i >= 2 & i <= 58:
                if j >= 2 & j <= 54 & check[k] < 0.25:
                    cnt = cnt + 1
                    Z = []
                    for yy in range(5):
                        ZZ = []
                        for xx in range(5):
                            ZZ.append(x_train[(i + xx - 2) + (j + yy - 2) * 61])
                        Z.append(ZZ)
                    X.append(Z)
                    cA, cD = pywt.dwt(y_train[i + j * 61], 'db6')
                    Y.append(y_train[i + j * 61])
        X = np.array(X)
        # print(X)
        print(np.shape(X))
        Y = np.array(Y)
        # print(Y)
        print(np.shape(Y))

        x_train, x_test, y1, y2 = train_test_split(X, Y, test_size=0.1, random_state=0)

        y_train = []
        y_test = []
        for i in range(len(y1)):
            cA, cD = pywt.dwt(y1[i], 'db6')
            kk = 0
            for j in range(29):
                kk = kk + cA[j]
            # if (cA[j] > kk):
            # kk = cA[j]
            # cA = cA / kk
            # y_train.append(y1[i])
            y_train.append(cA[gogogo])
        # y_train.append([cA[0],cA[1],cA[2],cA[3],cA[4],cA[5],cA[6],cA[7],cA[8],cA[9],cA[10],cA[11],cA[12],cA[13],cA[14],cA[15],cA[16],cA[17],cA[18],cA[19],cA[20],cA[21],cA[22],cA[23],cA[24],cA[25],cA[26],cA[27],cA[28]])
        for i in range(len(y2)):
            cA, cD = pywt.dwt(y2[i], 'db6')
            kk = 0
            for j in range(29):
                kk = kk + cA[j]
            # if (cA[j] > kk):
            # kk = cA[j]
            # cA = cA / kk
            # y_test.append(y2[i])
            y_test.append(cA[gogogo])
        # y_test.append([cA[0],cA[1],cA[2],cA[3],cA[4],cA[5],cA[6],cA[7],cA[8],cA[9],cA[10],cA[11],cA[12],cA[13],cA[14],cA[15],cA[16],cA[17],cA[18],cA[19],cA[20],cA[21],cA[22],cA[23],cA[24],cA[25],cA[26],cA[27],cA[28]])

        from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, MaxPooling2D
        from keras.models import Sequential
        from keras import backend as K

        K.set_image_dim_ordering('tf')
        model = Sequential()
        model.add(Conv2D(16,
                         activation='relu',
                         input_shape=(5, 5, qaq),
                         padding='valid',
                         nb_row=1,
                         nb_col=1))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, activation='relu',
                         padding='valid',
                         nb_row=3,
                         nb_col=3))
        model.add(Dropout(0.2))
        model.add(Conv2D(16, activation='relu',
                         padding='valid',
                         nb_row=1,
                         nb_col=1))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, activation='relu',
                         padding='valid',
                         nb_row=3,
                         nb_col=3))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        minn = 10000
        k = 10000
        mink = []
        mino = []
        minans = []

        print(np.shape(x_train))
        # print((x_train))
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        print(np.shape(y_train))
        # print((y_train))
        maxe = 10000
        my_anss = []
        my_ans_go = []
        for e in range(20000):

            model.fit(x_train, y_train, batch_size=32, epochs=1, validation_data=[x_test, y_test])
            ans = model.predict(x_test)
            mse = 0
            for i in range(len(ans)):
                mse = mse + (ans[i] - y_test[i]) ** 2
            mse = mse / len(ans)
            if mse < maxe:
                my_ans_go = ans
                maxe = mse
            # print(ans)
            # raw_input("\n")
            # print(x_test)
            A = []
            anss = 0
            n = 0
        # print(my_ans)
        print(np.shape(my_ans))
        print(np.shape(my_ans_go))

        # z = np.vstack(y_test)
        sum_i, kl_i, rmlse_i = entropy_evaluation("CNN", y_test, my_ans_go)
        # output = pandas.DataFrame(list(z))
        # output.to_csv(str(gogogo) + "focus" + str(qaq) +  " my_pred" + str(kl_i) + ".csv", index=False)

        for e in range(len(y_test)):
            my_ans[e][gogogo] = my_ans_go[e]
# z = np.vstack(y_test)

# output = pandas.DataFrame(list(z))
# output.to_csv(str(gogogo) + "aha" + str(each_ans) +  " my_pred" + str(each_max) + ".csv", index=False)

import pandas

real = []
pred = []

for i in range(len(my_ans)):
    pred.append(pywt.idwt(my_ans[i], None, 'db6'))
    real.append(y2[i])

real = np.array(real)
pred = np.array(pred)
print(np.shape(real))
print(np.shape(pred))
sums = []
kl = []
rmlse = []
all_sum = 0
all_kl = 0
all_rmlse = 0
for i in range(48):
    print(i)
    sum_i, kl_i, rmlse_i = entropy_evaluation("CNN", real[:, i], pred[:, i])
    sums.append(sum_i)
    kl.append(kl_i)
    rmlse.append(rmlse_i)
    all_sum = all_sum + sum_i ** 2

for i in range(48):
    all_kl = kl[i] * (sums[i] ** 2) / all_sum + all_kl
    all_rmlse = rmlse[i] * (sums[i] ** 2) / all_sum + all_rmlse

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print('CNN', 'KL Divergence: ', all_kl)
print('CNN', 'RMLSE: ', all_rmlse)

z = np.vstack((pred))

output = pandas.DataFrame(list(z))
output.to_csv("my_pred" + str(all_kl) + ".csv", index=False)

z = np.vstack(real)

output = pandas.DataFrame(list(z))
output.to_csv("my_real" + str(all_kl) + ".csv", index=False)

print(best)
print(bestn)
