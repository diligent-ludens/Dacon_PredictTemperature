import pandas as pd
import numpy as np
import random
import tensorflow as tf
import math
from datetime import datetime
import scipy.stats as sp
import matplotlib.pyplot as plt

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# 하이퍼 파라매터
learning_rate = 0.001
epoch1 = 30
batch_size1 = 128
epoch2 = 40
batch_size2 = 64
interval = 12
keep_prob = 0.2
output_size = 1

# 컬럼 선별
drop_col = [
    'X04', 'X10', 'X21', 'X36', 'X39',  # 강수량
    'X11', 'X14', 'X16', 'X19', 'X34',  # 일사량
    'X01', 'X06', 'X22', 'X27', 'X29',  # 현지기압
    'X05', 'X23', 'X33', 'X08', 'X09',  # 해면기압
    'X13', 'X15'  # 풍향
            ]

# 레이어 파라매터
lstm_size1 = 300
lstm_size2 = 286
l1_size = 400
l2_size = 300
l3_size = 150
l4_size = 100
l5_size = 75
l6_size = 50

lstm_size1_model2 = 160
l1_size_model2 = 100
l2_size_model2 = 105
l3_size_model2 = 36
l4_size_model2 = 45
l5_size_model2 = 30
l6_size_model2 = 10


'''
# 컬럼 항목별로 컬럼의 이름만 변수에 저장
temper = ['X00', 'X07', 'X28', 'X31', 'X32']
hum = ['X12', 'X20', 'X30', 'X37', 'X38']
raining = ['X04', 'X10', 'X21', 'X36', 'X39']
air_pressure = ['X01', 'X06', 'X22', 'X27', 'X29']
wind_speed = ['X02', 'X03', 'X18', 'X24', 'X26']
sea_pressure = ['X05', 'X08', 'X09', 'X23', 'X33']
sunrise = ['X11', 'X14', 'X16', 'X19', 'X34']
wind_way = ['X13', 'X15', 'X17', 'X25', 'X35']
'''

# 재생산성을 위해 시드 고정
np.random.seed(7)
random.seed(7)
tf.random.set_seed(7)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 데이터 전처리
def instant_data(df):
    update = []
    diff = 0.0

    for i in range(0, len(df), 1):
        if i > 0:
            diff = df.iloc[i] - df.iloc[i-1]
            if (diff < 0).any():
                diff = 0.0
        update.append(diff)

    update = pd.DataFrame(update)

    return update

def difference(df, list):
    result=pd.DataFrame()
    for i in list:
        temp = df[str(i)]
        temp = instant_data(temp)
        temp.columns=[str(i)]
        result = pd.concat([result,temp],axis=1)
    return result

# MinMaxScaler
def MinMaxScaler(data):
    numerator = data - np.min(data, axis=0)
    denominator = np.max(data, axis=0) - np.min(data, axis=0)

    return numerator / (denominator + 1e-7)


# 기상청 데이터만 추출
X_train = train.loc[:, 'X00':'X39']
test = test.loc[:, 'X00':'X39']

# Sunrise
# 0인 컬럼 3개 제외
filtering_sunrise = difference(train,["X11","X34"])
filtering_sunrise_t = difference(test,["X11","X34"])


# standardization 을 위해 평균과 표준편차 구하기
MEAN = X_train.mean()
STD = X_train.std()

# 표준편차가 0일 경우 대비하여 1e-07 추가
X_train = (X_train - MEAN) / (STD + 1e-07)
test = (test - MEAN) / (STD + 1e-07)

X_train[['X17', 'X25', 'X35']] = np.cos(math.pi*train[['X17', 'X25', 'X35']]/180)
test[['X17', 'X25', 'X35']] = np.cos(math.pi*test[['X17', 'X25', 'X35']]/180)


X_train = X_train.drop(drop_col, axis=1)
X_train = pd.concat([X_train, filtering_sunrise], axis=1)

test = test.drop(drop_col, axis=1)
test = pd.concat([test,  filtering_sunrise_t], axis=1)



X_train.to_csv('filtering_train.csv')

# RNN 모델에 입력 할 수 있는 시계열 형태로 데이터 변환
def convert_to_timeseries(df, interval):
    sequence_list = []
    target_list = []

    for i in range(df.shape[0] - interval):
        sequence_list.append(np.array(df.iloc[i:i + interval, :-1]))
        target_list.append(df.iloc[i + interval, -1])

    sequence = np.array(sequence_list)
    target = np.array(target_list)

    return sequence, target



y_columns = ['Y09', 'Y15', 'Y16']
# t시점 이전 120분의 데이터로 t시점의 온도를 추정할 수 있는 학습데이터 형성
sequence = np.empty((0, interval, 40-len(drop_col)+2))
target = np.empty((0,))
for column in y_columns:
    concat = pd.concat([X_train, train[column]], axis=1)

    _sequence, _target = convert_to_timeseries(concat.head(144 * 30), interval=interval)

    sequence = np.vstack((sequence, _sequence))
    target = np.hstack((target, _target))


# convert_to_timeseries 함수를 쓰기 위한 dummy feature 생성
X_train['dummy'] = 0
# train set에서 도출된 평균과 표준편차로 standardization 실시


# convert_to_timeseries 함수를 쓰기 위한 dummy feature 생성
test['dummy'] = 0
# train과 test 기간을 합쳐서 120분 간격으로 학습데이터 재구축
X_test, _ = convert_to_timeseries(pd.concat([X_train, test], axis = 0), interval=interval)


# test set 기간인 후반부 80일에 맞게 자르기
X_test = X_test[-144*80:, :, :]
# 만들어 두었던 dummy feature 제거
X_train.drop('dummy', axis = 1, inplace = True)
test.drop('dummy', axis = 1, inplace = True)

# fine tuning 할 때 사용할 학습데이터 생성 (Y18)
finetune_X, finetune_y = convert_to_timeseries(pd.concat([X_train.tail(432), train['Y18'].tail(432)], axis = 1), interval=interval)

# loss가 3미만으로 떨어지면 학습 종료 시키는 기능
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        if(logs.get('loss') < 3):
            print('\n Loss is under 3, cancelling training')
            self.model.stop_training = True
callbacks = myCallback()
c_back = 3
# 간단한 lstm 모델 구축하기

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(lstm_size1, input_shape=sequence.shape[-2:],return_sequences=True),
    tf.keras.layers.LSTM(lstm_size2, return_sequences=False),
    tf.keras.layers.Dense(l1_size, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l2_size, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l3_size, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l4_size, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l5_size, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l6_size, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(output_size)
])

simple_lstm_model2 = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(lstm_size1_model2, input_shape=sequence.shape[-2:],return_sequences=False),
    #tf.keras.layers.LSTM(lstm_size2_model2, return_sequences=False),
    tf.keras.layers.Dense(l1_size_model2, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l2_size_model2, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l3_size_model2, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l4_size_model2, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l5_size_model2, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(l6_size_model2, activation='linear', kernel_initializer='glorot_normal'),
    tf.keras.layers.Dense(output_size)
])


models = []
num_models = 2
finetune_pred=[]
models.append(simple_lstm_model)
models.append(simple_lstm_model2)

for i in range(num_models):
    models[i].compile(optimizer=tf.keras.optimizers.Adamax(learning_rate, beta_1=0.9, beta_2=0.999), loss='mse')

    # 모델 학습
    models[i].fit(
        sequence, target,
        epochs=epoch1,
        batch_size=batch_size1,
        verbose=2,
        shuffle=False,
        callbacks=[callbacks]
    )
    # LSTM 레이어는 고정
    models[i].layers[0].trainable = False
    #models[i].layers[1].trainable = False
    #models[i].layers[2].trainable = False

    # LSTM 레이어는 고정 시켜두고, DNN 레이어에 대해서 fine tuning 진행 (Transfer Learning)
    finetune_history = models[i].fit(
                finetune_X, finetune_y,
                epochs=epoch2,
                batch_size=batch_size2,
                shuffle=False,
                verbose = 2)

    # 예측하기
    finetune_pred.append(models[i].predict(X_test).reshape(1,-1)[0])


print(np.shape(finetune_pred))
finetune_pred = np.mean(finetune_pred,axis=0)
print(np.shape(finetune_pred))

# 제출 파일 만들기
submit = pd.DataFrame({'id': range(144*33, 144*113),
              'Y18': finetune_pred.reshape(1, -1)[0]})
submit.to_csv('baseline_result.csv', index=False)

stand = pd.read_csv("data/base.csv")
check1 = stand.iloc[1:, 1]
check2 = submit.iloc[1:, 1]
check = abs(check1.sub(check2))
check[check < 1] = 0
check = check.mul(check)
check = check.mean()
print(check)
submit = submit.describe()
print(submit)

#로그 파일 작성
now = datetime.now()
with open("history/log{}_{}_{}_({}).txt".format(str(now.day),str(now.hour),str(now.minute),str(round(finetune_history.history["loss"][-1],3))),"w") as log:
    log.writelines([
        "loss = ",str(finetune_history.history["loss"][-1]),
        "\n# 하이퍼 파라매터",
        "\nlearning_rate = ", str(learning_rate),
        "\nepoch1 = ", str(epoch1),
        "\nbatch_size1 = ", str(batch_size1),
        "\nepoch2 = ", str(epoch2),
        "\nbatch_size2 = ", str(batch_size2),
        "\ninterval_1 = ", str(interval),
        "\n레이어 파라매터",
        "\n모델의 수 : ", str(num_models), "\n",

        "\nlstm_size1 = ", str(lstm_size1),
        "\nlstm_size2 = ", str(lstm_size2),
        "\nl1_size = ", str(l1_size),
        "\nl2_size = ", str(l2_size),
        "\nl3_size = ", str(l3_size),
        "\nl4_size = ", str(l4_size),
        "\nl5_size = ", str(l5_size),
        "\nl6_size = ", str(l6_size), "\n",

        "\nlstm_size1_model2 = ", str(lstm_size1_model2),
        #"\nlstm_size2_model2 = ", str(lstm_size2_model2),
        "\nl1_size_model2 = ", str(l1_size_model2),
        "\nl2_size_model2 = ", str(l2_size_model2),
        "\nl3_size_model2 = ", str(l3_size_model2),
        "\nl4_size_model2 = ", str(l4_size_model2),
        "\nl5_size_model2 = ", str(l5_size_model2),
        "\nl6_size_model2 = ", str(l6_size_model2),

        "\noutput_size = ", str(output_size),
        "\ny_columns = ", str(y_columns),
        "\ndrop_x_columns = ", str(drop_col),
        "\nCallback = ", str(c_back)

    ])

#시각화
'''
plt.plot(test_predict, color="green")
plt.plot(testY, color="orange")
plt.show()
'''

plt.plot(finetune_pred.reshape(1, -1)[0], color="green")
plt.plot(stand.iloc[1:, 1], color="orange")
plt.show()
