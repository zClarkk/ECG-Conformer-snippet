# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from .custom import split_data_set
from .base import BaseDataset, BUFFER_SIZE
from ..utils.utils import preprocess_paths
from ..featurizers import SignalFeaturizer, ClassFeaturizer

AUTOTUNE = tf.data.experimental.AUTOTUNE
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
path_c0 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dataset_c0.txt')

class ECGDataset(BaseDataset):
    my_y = []
    fill_me = []
    flag_train_ds = True
    def __init__(self,
                 data_paths: List[str],
                 diagnostics_paths: List[str],
                 signal_featurizer: SignalFeaturizer,
                 class_featurizer: ClassFeaturizer,
                 header: bool = True,
                 cache: bool = False,   # was False
                 shuffle: bool = True,  # was True
                 buffer_size: int = BUFFER_SIZE,
                 drop_remainder: bool = True,
                 n_folds: int = 4, # was 5 or 4
                 name: str = 'train',
                 **kwargs):
        super().__init__(data_paths, cache=cache, shuffle=shuffle,
                         buffer_size=buffer_size, drop_remainder=drop_remainder, name=name, **kwargs)
        self.diagnostics_paths = preprocess_paths(diagnostics_paths)
        self.header = 0 if header else None
        self.signal_featurizer = signal_featurizer
        self.class_featurizer = class_featurizer
        self.total_steps = None
        self.data_files = None
        # self.kfolds = KFold(n_splits=n_folds)
        self.kfolds = StratifiedKFold(n_splits=n_folds)

    def read_entries(self):
        self.diagnostics = None
        for _diag_path in self.diagnostics_paths:
            _new_labels = pd.read_excel(_diag_path, index_col=0)
            if self.diagnostics is None: self.diagnostics = _new_labels
            else: self.diagnostics.append(_new_labels)

        ### MINE
            self.my_y = pd.read_excel(_diag_path, names=None, index_col=None, usecols = "B")
        tmp = self.my_y['Rhythm'].values.tolist()
        o = 0
        while o < len(tmp):
            if tmp[o] == 'AF':      self.fill_me.append(0)
            if tmp[o] == 'AFIB':    self.fill_me.append(1)
            if tmp[o] == 'SA':      self.fill_me.append(2)
            if tmp[o] == 'SB':      self.fill_me.append(3)
            if tmp[o] == 'SR':      self.fill_me.append(4)
            if tmp[o] == 'ST':      self.fill_me.append(5)
            if tmp[o] == 'SVT':     self.fill_me.append(6)
            o += 1
        ### MINE

        if self.diagnostics is None:
            raise ValueError('Unable to load diagnostic data, please check data paths')
        self.data_files = []
        for _dir in self.data_paths:
            self.data_files += tf.io.gfile.glob(os.path.join(_dir, '*.csv'))
        sorted = np.array(self.data_files)
        self.data_files = (np.sort(sorted))

    def generator(self, data_files):
        def fn():
            for _file in data_files:
                _data = pd.read_csv(_file, header=self.header, names=LEADS, encoding='utf-8', dtype=np.float32)
                _name = os.path.basename(os.path.splitext(_file)[0])
                _labels = {_col.lower(): str(self.diagnostics.loc[_name, _col]) for _col in list(self.diagnostics.columns)}
                _inputs = {_lead: _data.loc[:, _lead].to_numpy() for _lead in LEADS}
                yield _inputs, _labels
        return fn

    def parse(self, inputs, labels):
        with tf.device('/CPU:0'):
            _tinputs = {_lead: tf.convert_to_tensor(inputs[_lead], dtype=tf.float32) for _lead in LEADS}
            _tlabels = {_col.lower(): tf.convert_to_tensor(labels[_col.lower()]) for _col in list(self.diagnostics.columns)}
            return self.signal_featurizer.extract(_tinputs), self.class_featurizer.extract(_tlabels)

    def create_dataset(self, generator, batch_size):
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_shapes=(
                {_lead: tf.TensorShape([None]) for _lead in LEADS},
                {_col.lower(): tf.TensorShape([]) for _col in list(self.diagnostics.columns)}
            ),
            output_types=(
                {_lead: tf.float32 for _lead in LEADS},
                {_col.lower(): tf.string for _col in list(self.diagnostics.columns)}
            )
        )
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        if self.cache: dataset = dataset.cache()
        if self.shuffle: dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=False)

        if self.flag_train_ds:
            ### Split data into datasets for each label
            c0_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([0], dtype=tf.int64))
            c1_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([1], dtype=tf.int64))
            c2_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([2], dtype=tf.int64))
            c3_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([3], dtype=tf.int64))
            c4_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([4], dtype=tf.int64))
            c5_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([5], dtype=tf.int64))
            c6_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([6], dtype=tf.int64))

            ### Adjust amount of data for each dataset
            c0_ds = c0_ds.repeat(10).take(2916)      # AF
            c1_ds = c1_ds.repeat(10).take(1)     # AFIB
            c2_ds = c2_ds.repeat(10).take(2916)      # SA
            c3_ds = c3_ds.repeat(10).take(2916)     # SB
            c4_ds = c4_ds.repeat(10).take(1)     # SR
            c5_ds = c5_ds.repeat(10).take(2916)     # ST
            c6_ds = c6_ds.repeat(10).take(2916)      # SVT
            ### Concatenate datasets into one
            cds1 = c0_ds.concatenate(c1_ds)
            cds2 = cds1.concatenate(c2_ds)
            cds3 = cds2.concatenate(c3_ds)
            cds4 = cds3.concatenate(c4_ds)
            cds5 = cds4.concatenate(c5_ds)
            cds6 = cds5.concatenate(c6_ds)
            ### Change all labels to this label
            # cds6 = cds6.map(lambda x, y: (x := x, y := tf.constant([0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)))
            # cds6 = cds6.concatenate(c0_ds)
            # cds6 = c0_ds.concatenate(c3_ds)
            dataset = cds6.shuffle(25000, reshuffle_each_iteration=False)
        # if self.flag_train_ds:
        # ### Split data into datasets for each label
        #     c0_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([0], dtype=tf.int64))
        #     c1_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([1], dtype=tf.int64))
        #     c2_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([2], dtype=tf.int64))
        #     c3_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([3], dtype=tf.int64))
        #     c4_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([4], dtype=tf.int64))
        #     c5_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([5], dtype=tf.int64))
        #     c6_ds = dataset.filter(lambda x, y: tf.argmax(y) == tf.constant([6], dtype=tf.int64))
        #
        #     ### Adjust amount of data for each dataset
        #     c0_ds = c0_ds.repeat(10).take(250)
        #     c1_ds = c1_ds.repeat(10).take(250)
        #     c2_ds = c2_ds.repeat(10).take(250)
        #     c3_ds = c3_ds.repeat(10).take(250)
        #     c4_ds = c4_ds.repeat(10).take(250)
        #     c5_ds = c5_ds.repeat(10).take(1500)
        #     c6_ds = c6_ds.repeat(10).take(250)
        #     ### Concatenate datasets into one
        #     #cds1 = c0_ds.concatenate(c1_ds)
        #     cds2 = c1_ds.concatenate(c3_ds)
        #     cds3 = cds2.concatenate(c0_ds)
        #     cds4 = cds3.concatenate(c4_ds)
        #     cds5 = cds4.concatenate(c2_ds)
        #     cds6 = cds5.concatenate(c6_ds)
        #     ### Change all labels to this label
        #     cds6 = cds6.map(lambda x, y: (x := x, y := tf.constant([0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)))
        #     cds6 = cds6.concatenate(c5_ds)
        #     dataset = cds6.shuffle(25000, reshuffle_each_iteration=False)
        ### Combines consecutive elements of this dataset into padded batches.
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(self.signal_featurizer.tensorshape, self.class_featurizer.tensorshape),
            drop_remainder=self.drop_remainder
        )
        # dataset = dataset.batch(batch_size=batch_size, drop_remainder = self.drop_remainder)

        ### Most dataset input pipelines should end with a call to prefetch. This allows later elements to be prepared
        # while the current element is being processed. This often improves latency and throughput, at the cost of
        # using additional memory to store prefetched elements.
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def split(self):
        if self.data_files is None: self.read_entries()
        # for train_indices, test_indices in self.kfolds.split(self.data_files):
        for train_indices, test_indices in self.kfolds.split(self.data_files, self.fill_me):
            train_data_files = self.data_files[train_indices]
            test_data_files = self.data_files[test_indices]
            yield train_data_files, test_data_files


    def create(self, batch_size=None):
        if batch_size is None: batch_size = 1
        for train_data_files, test_data_files in self.split():
            train_dataset = self.create_dataset(self.generator(train_data_files), batch_size=batch_size)
            self.flag_train_ds = False
            test_dataset = self.create_dataset(self.generator(test_data_files), batch_size=batch_size)
            self.flag_train_ds = True
            yield train_dataset, test_dataset
