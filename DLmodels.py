from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Flatten, TimeDistributed, Input, Permute, Conv1D, Bidirectional, RepeatVector,Multiply

class Generate_model():#为所有模型创建一个通用class
    def __init__(self):
        self.model = Sequential()
    def generate_lstm_model(self,n_input, n_out, n_features):#LSTM模型
        self.model = Sequential()
        self.model.add(LSTM(64, activation='relu',  input_shape=(n_input, n_features)))#64个神经元，激活函数为relu
        self.model.add(Dropout(0.1))#丢失层，丢失率为0.1
        self.model.add(Dense(n_out))
        self.model.summary()
        # 模型编译
        self.model.compile(loss="mse", optimizer='adam')
        return self.model

    def generate_seq2seq_model(self,n_input, n_out, n_features):
        self.model = Sequential()
        self.model.add(LSTM(128,input_shape=(n_input, n_features)))
        self.model.add(Dense(10, activation="relu"))
        # 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
        self.model.add(RepeatVector(1))#此为步长
        # Decoder(第二个 LSTM)
        self.model.add(LSTM(128,return_sequences=True))
        # TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
        self.model.add(TimeDistributed(Dense(units=n_out, activation="linear")))
        self.model.add(Flatten())#扁平层将（None,1,8)变为（None,1*8)
        self.model.summary()
        self.model.compile(loss="mse", optimizer='adam')
        return self.model


    #   注意力模块，主要是实现对step维度的注意力机制
    #   先Permute再进行注意力机制的施加，对齐输入输出的维度。
    #   需要首先将step维度转到最后一维，然后再进行全连接，根据每一个step的特征获得注意力机制的权值。
    def attention_block(self,inputs,time_step):
        # batch_size, time_steps, lstm_units = batch_size, lstm_units, time_steps
        a = Permute((2, 1))(inputs)
        # batch_size, lstm_units, time_steps -> batch_size, lstm_units, time_steps
        a = Dense(time_step, activation='softmax')(a)#和步长有关
        # batch_size, lstm_units, time_steps = batch_size, time_steps, lstm_units
        a_probs = Permute((2, 1), name='attention_vec')(a)
        # 相当于获得每一个step中，每个特征的权重
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul



    def cnn_lstm_attention_model(self, n_input, n_out, n_features):
        inputs = Input(shape=(n_input, n_features))
        x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs) #一维卷积层
        x = Dropout(0.3)(x)
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(x)#经实验Bidirectional LSTM的效果要比LSTN效果好
        lstm_out = Dropout(0.3)(lstm_out)
        attention_mul = self.attention_block(lstm_out, n_input)
        attention_mul = Flatten()(attention_mul)
        output = Dense(n_out, activation='sigmoid')(attention_mul)
        model = Model(inputs=[inputs], outputs=output)
        model.summary()
        model.compile(loss="mse", optimizer='adam')
        return model

    def SARIMAX(self,data):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        df1 = data[["meantemp", "humidity", "wind_speed"]]

        train_size = int(len(df1) * 0.8)
        train, test = df1.iloc[:train_size], df1.iloc[train_size:]

        # 设定参数
        order = (1, 1, 6)  # 非季节参数 (p, d, q)
        seasonal_order = (1, 1, 1, 7)  # 季节参数 (P, D, Q, S)

        # 训练模型
        sarima_model = SARIMAX(endog=train['meantemp'], exog=train[['humidity', 'wind_speed']],
                               order=order, seasonal_order=seasonal_order)
        sarima_model_fit = sarima_model.fit()
        return sarima_model_fit
