import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from .fada_dataset  import FadaDataset, Generator


class FadaModel:
    def __init__(self, model):
        """
        :param model: Deep learning model to be used as the source model for domain adaptation
        """
        self.model = model
        self.model_source = tf.keras.models.clone_model(self.model)
        self.model_target = tf.keras.models.clone_model(self.model)
        self.model_fada = self._create_model()

    def _create_model(self):

        self.model_source.set_weights(self.model.get_weights())
        self.model_target.set_weights(self.model.get_weights())

        for layer in self.model_source.layers:
            layer._name = layer.name + str("_source")

        for layer in self.model_target.layers:
            layer._name = layer.name + str("_target")

        features_DCD = tf.keras.layers.concatenate([self.model_source.get_layer("features_source").output,
                                                    self.model_target.get_layer("features_target").output], axis=1)
        fc_1_DCD = Dense(128, activation="relu", use_bias=True, name="fc_1_DCD")(features_DCD)
        drop_1_DCD = Dropout(0.5)(fc_1_DCD)
        out_DCD = Dense(4, activation="softmax", use_bias=True, name="out_DCD")(drop_1_DCD)

        model_fada = tf.keras.Model(inputs=[self.model_source.input,  self.model_target.input],
                                    outputs=[self.model_source.output, self.model_target.output, out_DCD])

        print(model_fada.summary())
        return model_fada

    def get_model(self):
        return self.model_fada

    def fit(self, x_source, y_source, x_target, y_target, samples_source=10, samples_target=5):
        print("Preparing fada dataset generators ...")
        dataset_object = FadaDataset(x_source, y_source, x_target, y_target, samples_source, samples_target)
        paired_gen = Generator(dataset_object.get_fada_dataset(), batch_size=10)
        paired_gen_confusion = Generator(dataset_object.get_confusion_dataset(), batch_size=10)

        for layer in self.model_fada.layers:
            if "DCD" in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False

        print("The Domain-Class Discriminator will be trained...")
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, min_delta=0.005, mode="min")
        self.model_fada.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
                                loss={"out_DCD": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      "out_source": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      "out_target": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
                                loss_weights={"out_DCD": 1, "out_source": 0.2, "out_target": 0.1}, metrics=["accuracy"])

        self.model_fada.fit(paired_gen, verbose=1, epochs=200, callbacks=[callback])

        print("Training the DCD classifier is done")

        print("Training model layers ...")

        for layer in self.model_fada.layers:
            if "DCD" in layer.name:
                print(layer.name, " will NOT be trained")
                layer.trainable = False
            else:
                layer.trainable = True

        self.model_fada.compile(optimizer=tf.keras.optimizers.Adam(0.00001),
                                loss={"out_DCD": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      "out_source": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      "out_target": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
                                loss_weights={"out_DCD": 1, "out_source": 0.2, "out_target": 1}, metrics=["accuracy"])

        self.model_fada.fit(paired_gen_confusion, verbose=1, epochs=200, callbacks=[callback])

        print("training FADA is done")

        return self.model_target

