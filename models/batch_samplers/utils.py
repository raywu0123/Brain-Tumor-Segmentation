from itertools import chain


def flatten(listOfLists):
    "Flatten one level of nesting"
    return list(chain.from_iterable(listOfLists))


def normalize_position(batch_indexes, batch_label_shape):
    dhw = batch_label_shape[-3:]
    normalized_position = batch_indexes[:, 1:] / dhw
    return normalized_position
