import tensorflow as tf
from tensorflow.keras.layers import Flatten, MaxPool2D, Dropout, Input, Dense, Conv2D, LeakyReLU, BatchNormalization




class CnnModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.create_model()

    def create_model(self):
        input = Input(self.input_shape, name="input")
        d = Conv2D(filters=32, kernel_size=(4, 4), padding="same", activation="relu", name="c1")(input)
        d = MaxPool2D(pool_size=(4, 4), strides=(2, 2), padding="Valid", name="m1")(d)
        d = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", name="c2")(d)
        d = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="Valid", name="m2")(d)
        features = Flatten(name="features")(d)

        fc_4 = Dense(64, activation="relu", use_bias=True, name="fc",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(features)
        drop_4 = Dropout(0.5)(fc_4)
        out_7 = Dense(10, use_bias=True, name="out")(drop_4)

        model = tf.keras.Model(inputs=input, outputs=out_7)

        print(model.summary())

        return model

    def fit(self, x_train, y_train, x_test, y_test):



        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=["accuracy"])
        self.model.fit(x_train, y_train, verbose=1,
                       epochs=10)

        self.model.evaluate(x_test, y_test)

    def evaluate(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)

