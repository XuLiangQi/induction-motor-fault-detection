from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt

opt = Adam(learning_rate=0.001)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

class LSTMModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        LSTMmodel = Sequential()
        LSTMmodel.add(LSTM(300, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
        LSTMmodel.add(Dropout(0.5))
        LSTMmodel.add(LSTM(300, activation='relu'))
        LSTMmodel.add(Dropout(0.5))
        LSTMmodel.add(Dense(8, activation='softmax'))
        self.model = LSTMmodel

    def train(self):
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.hist = self.model.fit(self.X_train, self.y_train, epochs=100, callbacks=[early_stop], validation_data=(self.X_test, self.y_test), batch_size=int(len(self.X_train)/2))
        self.model.summary()

    def predict(self, X_val):
        y_pred = self.model.predict(X_val)
        return y_pred

    def plot(self):
        plt.plot(self.hist.history['accuracy'])
        plt.plot(self.hist.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()