
from utils.preprocessing import *
from fada.fada_model import *
from cnn.cnn_model import *
import argparse
import tensorflow as tf

def main():
    #Loading MNIST dataset as source domain
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = LoadDataset.normalize(x_train)
    x_test = LoadDataset.normalize(x_test)
    #Loading USPS dataset as target domain
    x_tr, y_tr, x_te, y_te = LoadDataset.target_dataset()

    input_shape = x_train[0].shape
    cnn_model = CnnModel(input_shape)
    cnn_model.fit(x_train, y_train, x_test, y_test)

    print("The model has the following accuracy in the target domain BEFORE domain adaptation:")
    cnn_model.evaluate(x_tr, y_tr)

    fada_model = FadaModel(cnn_model.model)
    target_model = fada_model.fit(x_train, y_train, x_tr, y_tr, args.n_source, args.n_target)

    target_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    print("The model has the following accuracy in the target domain AFTER domain adaptation:")
    target_model.evaluate(x_tr, y_tr)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_source", default=10, type=int, help="number of source samples")
    ap.add_argument("--n_target", default=5, type=int, help="number of target samples")
    args = ap.parse_args()
    main()