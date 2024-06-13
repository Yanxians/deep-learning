import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import DLmodels
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#处理数据，训练模型

print('开始数据处理...')
# 加载数据
data = pd.read_csv('C:\\Users\yanran\Desktop\AI-report\dataset\DailyDelhiClimateTrain.csv')
data = data[["date","meantemp","humidity","wind_speed"]]
date_split = data['date'].str.split('/', expand=True)
date_split.columns = ['year', 'month', 'day']
data = pd.concat([data.drop('date', axis=1),date_split], axis=1)

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
# 划分特征和标签
def create_dataset(data, n_input):
    X, y = [], []
    for i in range(len(data) - n_input):
        X.append(data[i:i + n_input, 1:])
        y.append(data[i + n_input, 0])
    return np.array(X), np.array(y)

n_input = 12  # 输入时间步数，可调整
X, y = create_dataset(data_scaled, n_input)

# 划分训练集和测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
split2 = int(0.75*len(X_train))
X_train, X_val = X_train[:split2], X_train[split2:]
y_train, y_val = y_train[:split2], y_train[split2:]

# 训练模型
generator = DLmodels.Generate_model()

print('SARIMAX模型训练中...')
# SARIMAX模型
SARIMA = generator.SARIMAX(data)

print("LSTM模型训练中...")
# LSTM模型
lstm_model = generator.generate_lstm_model(n_input, 1, X.shape[2])
lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))
# lstm_model.save('')
# lstm_model = load_model('C:\\Users\yanran\Desktop\AI-report\models\\lstm4.keras')

print('Seq2seq模型训练中...')
# Seq2Seq模型
seq2seq_model = generator.generate_seq2seq_model(n_input, 1, X.shape[2])
seq2seq_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))
#seq2seq_model.save('')
# seq2seq_model = load_model('C:\\Users\yanran\Desktop\AI-report\models\\seq4.keras')

print('CNN+LSTM+ATTENTION模型训练中...')
#CNN + LSTM + 注意力模型
cnn_lstm_attention_model = generator.cnn_lstm_attention_model(n_input, 1, X.shape[2])
cnn_lstm_attention_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))
#cnn_lstm_attention_model.save('')
# cnn_lstm_attention_model = load_model('C:\\Users\yanran\Desktop\AI-report\models\\cla4.keras')
# 预测函数
def plot_predictions(model, X_test, y_test, scaler,model_name):
    predictions = model.predict(X_test)

    # 创建扩展的 predictions 数组
    predictions_expanded = np.zeros((predictions.shape[0], data_scaled.shape[1]))
    predictions_expanded[:, 0] = predictions[:, 0]  # 只反向缩放第一个特征（meantemp）

    # 反向缩放预测值和真实值
    predictions_scaled = scaler.inverse_transform(predictions_expanded)[:, 0]
    y_test_expanded = np.zeros((y_test.shape[0], data_scaled.shape[1]))
    y_test_expanded[:, 0] = y_test
    y_test_scaled = scaler.inverse_transform(y_test_expanded)[:, 0]

    plt.figure(figsize=(10, 5))
    plt.plot(predictions_scaled,color = 'red', label='Predictions')
    plt.plot(y_test_scaled, color = 'blue',label='True Values')
    mse = mean_squared_error(y_test_scaled, predictions_scaled)
    #打印 MSE到 plot上
    plt.text(0.01, 0.95, f'mse = {mse:.2f}', transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.title(f'{model_name} Predictions vs True Values')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()

print("开始绘图...")
# SARIMAX模型预测结果(SARIMAX为统计学模型，有些包无法与 Keras共享，故单独编程实现）
df1 = data[["meantemp", "humidity", "wind_speed"]]
train_size = int(len(df1) * 0.8)
test = df1.iloc[train_size:]
sarima_pred = SARIMA.predict(start=test.index[0], end=test.index[-1],
                                               exog=test[['humidity', 'wind_speed']])

plt.figure(figsize=(10, 5))
plt.plot(test.index, test['meantemp'], color = 'blue', label='True Values')
plt.plot(test.index, sarima_pred, color='red', label='Predictions')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('SARIMA Predictions VS True Values')
mse = mean_squared_error(test['meantemp'], sarima_pred)
plt.text(0.01, 0.95, f'mse = {mse:.2f}', transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.legend()
plt.show()

# LSTM模型预测结果
plot_predictions(lstm_model, X_test, y_test, scaler,"LSTM")

# Seq2Seq模型预测结果
plot_predictions(seq2seq_model, X_test, y_test, scaler,"Seq2seq")

#CNN + LSTM + 注意力模型预测结果
plot_predictions(cnn_lstm_attention_model, X_test, y_test, scaler,"CNN-LSTM-ATTENTION")
