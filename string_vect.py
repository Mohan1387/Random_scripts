"""
The main code for the Strings-to-Vectors assignment. See README.md and Instructions for details.
"""
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.index_vocab = {}
        self.start = start
        self.vocab = vocab
        idx = self.start
        if isinstance(self.vocab, list):
            for i in self.vocab:
                if i not in self.index_vocab.values() and idx not in self.index_vocab.keys():
                    self.index_vocab[idx] = i
                    idx += 1
        elif isinstance(self.vocab, str):
            for i in list(self.vocab):
                if i not in self.index_vocab.values() and idx not in self.index_vocab.keys():
                    self.index_vocab[idx] = i
                    idx += 1
        

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """
        size = len(object_seq)
        lst_arr = []
        swapped_dict = {value: key for key, value in self.index_vocab.items()}
        for el in object_seq:
            try:
                lst_arr.append(swapped_dict[el])
            except KeyError:
                lst_arr.append(self.start - 1)
                
        return np.array(lst_arr)


    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """
        ##YOUR CODE HERE##
        lst_arr = []
        max_col_len = 0
        for i in object_seq_seq:
            col_len = len(i)
            if col_len > max_col_len:
                max_col_len = col_len
            lst_arr.append(self.objects_to_indexes(i))

        return np.array([np.pad(arr, (0, max_col_len - len(arr)), 'constant', constant_values=self.start -1) for arr in lst_arr])

        

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        ##YOUR CODE HERE##
        arr = np.zeros(max(self.index_vocab.keys())+1)
        if len(object_seq) > 0:
            idx = self.objects_to_indexes(object_seq)
            arr[idx] = 1
            return arr
        else:
            return arr

    
    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """
        ##YOUR CODE HERE##
        arr = []
        for i in object_seq_seq:
            arr.append(self.objects_to_binary_vector(i))
            
        return np.array(arr)


    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        ##YOUR CODE HERE
        seq = []
        for i in index_vector:
            if i in self.index_vocab.keys():
                seq.append(self.index_vocab[i])
        return seq
    

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        ##YOUR CODE HERE##
        lst_arr = []
        for i in index_matrix:
            lst_arr.append(self.indexes_to_objects(i))
        return lst_arr
        

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        ##YOUR CODE HERE##
        return self.indexes_to_objects(np.flatnonzero(vector == 1)) 
        

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        ##YOUR CODE HERE##
 
        arr = []
        for i in binary_matrix:
            arr.append(self.binary_vector_to_objects(i))
        return arr

#--------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""stv_nn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1di_slS0XadBFCg4R7LPJVkG35o8lzsgw
"""

"""
The main code for the Strings-to-Vectors assignment. See README.md and Instructions for details.
"""
from typing import Sequence, Any

import numpy as np


from typing import Sequence, Any
import numpy as np


class Index:
    def __init__(self, vocab: Sequence[Any], start: int = 0):
        """
        Maps unique items from `vocab` to integers starting from `start`.
        """
        self.start = start
        self.index_vocab = {i: obj for i, obj in enumerate(dict.fromkeys(vocab), start)}
        self.vocab_to_index = {v: k for k, v in self.index_vocab.items()}

    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Converts a sequence of objects to their respective indices.
        Unknown objects get index `start - 1`.
        """
        default_index = self.start - 1
        return np.array([self.vocab_to_index.get(obj, default_index) for obj in object_seq])

    def objects_to_index_matrix(self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Converts sequences of sequences into a padded matrix of indices.
        """
        default_index = self.start - 1
        max_len = max(len(seq) for seq in object_seq_seq)
        return np.array([
            np.pad(self.objects_to_indexes(seq), (0, max_len - len(seq)), constant_values=default_index)
            for seq in object_seq_seq
        ])

    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Creates a binary vector with 1s at indices corresponding to object_seq.
        """
        size = max(self.index_vocab.keys()) + 1 if self.index_vocab else 0
        vec = np.zeros(size, dtype=int)
        for idx in self.objects_to_indexes(object_seq):
            if 0 <= idx < size:
                vec[idx] = 1
        return vec

    def objects_to_binary_matrix(self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Converts sequences of sequences to a binary matrix.
        """
        return np.array([self.objects_to_binary_vector(seq) for seq in object_seq_seq])

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Converts a vector of indices back to objects, skipping unknown indices.
        """
        return [self.index_vocab[i] for i in index_vector if i in self.index_vocab]

    def index_matrix_to_objects(self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Converts a matrix of indices back to a sequence of sequences of objects.
        """
        return [self.indexes_to_objects(row) for row in index_matrix]

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Converts a binary vector to the list of objects at positions where value is 1.
        """
        return self.indexes_to_objects(np.flatnonzero(vector))

    def binary_matrix_to_objects(self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Converts a binary matrix to a sequence of sequences of objects.
        """
        return [self.binary_vector_to_objects(row) for row in binary_matrix]

import sys

import numpy as np
import pytest

import stv_nn
def test_large_sequences():
    vocab = [chr(i) for i in range(sys.maxunicode + 1)]
    objects = list('schön día \U0010ffff' * 100)

    index = stv_nn.Index(vocab)
    vector = index.objects_to_binary_vector(objects)
    assert vector[ord('í')] == 1
    assert vector[ord('\U0010ffff')] == 1
    assert vector[ord('o')] == 0
    assert index.binary_vector_to_objects(vector) == list(' acdhnsíö\U0010ffff')
