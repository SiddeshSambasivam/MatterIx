import os
import gzip
import requests
from typing import Tuple

import numpy as np
from tqdm import tqdm

from .utils import to_categorical


def parseFileName(url: str) -> str:
    return url.split("/")[-1]


def parseDataset(f_path: str, size: int):

    reader = gzip.open(f_path, "rb")
    reader.read(
        16
    )  # first 16 bits contains the magic number and the dimension of the images

    data_buffer = reader.read((28 * 28 * int(size)))

    data = np.frombuffer(data_buffer, dtype=np.uint8).astype(np.float32)
    data = data.reshape(size, 784)

    return data


def parseLabels(f_path: str, size: int):

    reader = gzip.open(f_path, "rb")
    reader.read(8)
    labels = []

    buf = reader.read(size)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64).reshape(size, 1)

    return labels


def getData(url: str, data_type: str, size: int):
    """
    Returns the data as numpy array

    Parameter
    ---------
    Arg: url (str)
        Link to the dataset
    Arg: data (str)
        "images" or "labels"
    Arg: size (int)
        size of the dataset to read

    """

    f_path = parseFileName(url)

    if not os.path.exists(f_path):

        response = requests.get(url)

        with open(f_path, "wb") as file:
            file.write(response.content)

    if data_type == "images":
        data = parseDataset(f_path, size=size)

    elif data_type == "labels":
        data = parseLabels(f_path, size=size)

    return data


def getMNIST() -> Tuple[np.ndarray]:

    URLS = [
        (
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            60000,
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            10000,
        ),
    ]

    data = list()

    for img_url, label_url, size in tqdm(URLS):
        imgs = getData(img_url, "images", size=size)
        lbl = getData(label_url, "labels", size=size)
        data.extend([imgs, lbl])

    x_train, y_train, x_test, y_test = data

    x_train = x_train.astype("float32")
    x_train /= 255.0

    y_train = to_categorical(y_train)

    x_test = x_test.astype("float32")
    x_test /= 255.0

    y_test = to_categorical(y_test)

    return [x_train, y_train, x_test, y_test]
