import numpy as np
from tensorflow.keras.utils import Sequence


class FadaDataset:
    def __init__(self, x_source, y_source, x_target, y_target, number_source_samples, number_target_samples,
                 shuffle=True):
        self.x_source = x_source
        self.y_source = y_source
        self.x_target = x_target
        self.y_target = y_target
        self.n_target = number_target_samples
        self.n_source = number_source_samples
        self.shuffle = shuffle
        self.paired_dataset = self._create_fada_dataset()
        self.confusion_dataset = self._create_fada_confusion()

    def _create_fada_dataset(self):
        num_classes = 10
        paired_dataset = []

        # group 2: pairs of same class from different domain
        label = 1
        for i in range(num_classes):
            x_source, y_source = self._select_data(self.x_source, self.y_source, i)
            x_target, y_target = self._select_data(self.x_target, self.y_target, i)
            for s in range(self.n_source):
                for t in range(self.n_target):
                    paired_dataset.append((x_source[s], y_source[s],
                                           x_target[t], y_target[t], label))

        print("25% ...")
        # group1: pairs of same class from source domain
        label = 0
        size = len(paired_dataset)
        for j in range(num_classes):
            i = 0
            x_source, y_source = self._select_data(self.x_source, self.y_source, j)
            length = y_source.shape[0]
            while i < size // 10:
                index = np.random.randint(length, size=2)
                paired_dataset.append((x_source[index[0]], y_source[index[0]],
                                       x_source[index[1]], y_source[index[1]], label))
                i += 1

        print("50% ...")

        # group 3: pairs of different classes from source domain
        label = 2
        for _ in range(size):
            label_class = np.random.randint(10, size=2)
            if label_class[0] == label_class[1]:
                label_class = np.random.randint(10, size=2)

            x_source_1, y_source_1 = self._select_data(self.x_source, self.y_source, label_class[0])
            x_source_2, y_source_2 = self._select_data(self.x_source, self.y_source, label_class[1])
            length_1 = y_source_1.shape[0]
            length_2 = y_source_2.shape[0]
            index_1 = np.random.randint(length_1)
            index_2 = np.random.randint(length_2)
            paired_dataset.append((x_source_1[index_1], y_source_1[index_1],
                                   x_source_2[index_2], y_source_2[index_2], label))

        print("75% ...")

        #     #group 4:pairs of different domain and class
        label = 3
        for _ in range(size):
            label_class = np.random.randint(5, size=2)
            if label_class[0] == label_class[1]:
                label_class = np.random.randint(5, size=2)
            x_source, y_source = self._select_data(self.x_source, self.y_source, label_class[0])
            x_target, y_target = self._select_data(self.x_target, self.y_target, label_class[1])
            length_1 = y_source.shape[0]
            length_2 = y_target.shape[0]
            index_1 = np.random.randint(length_1)
            index_2 = np.random.randint(length_2)
            paired_dataset.append((x_source[index_1], y_source[index_1],
                                   x_target[index_2], y_target[index_2], label))

        print("##--done--##")
        if self.shuffle:
            np.random.shuffle(paired_dataset)
        print(len(paired_dataset))
        return paired_dataset

    def _create_fada_confusion(self):
        dataset = self.paired_dataset.copy()
        for i in range(len(dataset) - 1, -1, -1):
            label = dataset[i][4]
            if label == 2 or label == 0:
                dataset.pop(i)
            else:
                tmp = list(dataset[i])
                tmp[4] = (tmp[4] * -1) + 4
                dataset[i] = tuple(tmp)
        print(len(dataset))
        return dataset

    def _select_data(self, x, y, label):
        ind = np.where(y == label)
        return x[ind], y[ind]

    def get_fada_dataset(self):
        return self.paired_dataset

    def get_confusion_dataset(self):
        return self.confusion_dataset


class Generator(Sequence):

    def __init__(self, data: list, batch_size: int):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil((len(self.data) / self.batch_size)))

    def __getitem__(self, idx):
        batch_data = np.array(self.data[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_source = np.stack(batch_data[:, 0])
        batch_target = np.stack(batch_data[:, 2])
        batch_label_source = batch_data[:, 1]
        batch_label_target = batch_data[:, 3]
        batch_label_dcd = batch_data[:, 4]

        return ({"input_source": batch_source, "input_target": batch_target},
                {"out_source": batch_label_source.astype(np.float32),
                 "out_target": batch_label_target.astype(np.float32),
                 "out_DCD": batch_label_dcd.astype(np.float32)})
