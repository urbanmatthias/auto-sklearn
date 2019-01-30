import os
import numpy as np
from scipy.sparse import csr_matrix

class AutoMlReader(object):
    def __init__(self, path_to_info):
        self.num_entries = None
        self.num_features = None
        self.num_classes = None

        self.file_name = path_to_info
        self.data = None
        self.X = None
        self.Y = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        self.is_classification = None
        self.categorical_features = None
        self.is_multilabel = None
        self.max_runtime = None
        self.metric = None

    
    def read(self, **kwargs):
        path_to_info = self.file_name
        info_dict = dict()

        # read info file
        with open(path_to_info, "r") as f:
            for line in f:
                info_dict[line.split("=")[0].strip()] = line.split("=")[1].strip().strip("'")
        self.is_classification = "classification" in info_dict["task"]
        
        name = info_dict["name"]
        path = os.path.dirname(path_to_info)
        self.is_multilabel = "multilabel" in info_dict["task"] if self.is_classification else None
        self.metric = info_dict["metric"]
        self.max_runtime = float(info_dict["time_budget"])

        target_num = int(info_dict["target_num"])
        feat_num = int(info_dict["feat_num"])
        train_num = int(info_dict["train_num"])
        valid_num = int(info_dict["valid_num"])
        test_num = int(info_dict["test_num"])
        is_sparse = bool(int(info_dict["is_sparse"]))
        feats_binary = info_dict["feat_type"].lower() == "binary"

        # read feature types
        force_categorical = []
        force_numerical = []
        if info_dict["feat_type"].lower() == "binary" or info_dict["feat_type"].lower() == "numerical":
            force_numerical = [i for i in range(feat_num)]
        elif info_dict["feat_type"].lower() == "categorical":
            force_categorical = [i for i in range(feat_num)]
        elif os.path.exists(os.path.join(path, name + "_feat.type")):
            with open(os.path.join(path, name + "_feat.type"), "r") as f:
                for i, line in enumerate(f):
                    if line.strip().lower() == "numerical":
                        force_numerical.append(i)
                    elif line.strip().lower() == "categorical":
                        force_categorical.append(i)
        
        # read data files
        reading_function = self.read_datafile if not is_sparse else (
            self.read_sparse_datafile if not feats_binary else self.read_binary_sparse_datafile)
        self.X = reading_function(os.path.join(path, name + "_train.data"), (train_num, feat_num))
        self.Y = self.read_datafile(os.path.join(path, name + "_train.solution"), (train_num, target_num))

        if os.path.exists(os.path.join(path, name + "_valid.data")) and \
            os.path.exists(os.path.join(path, name + "_valid.solution")):
            self.X_valid = reading_function(os.path.join(path, name + "_valid.data"), (valid_num, feat_num))
            self.Y_valid = self.read_datafile(os.path.join(path, name + "_valid.solution"), (valid_num, target_num))
        
        if os.path.exists(os.path.join(path, name + "_test.data")) and \
            os.path.exists(os.path.join(path, name + "_test.solution")):
            self.X_test = reading_function(os.path.join(path, name + "_test.data"), (test_num, feat_num))
            self.Y_test = self.read_datafile(os.path.join(path, name + "_test.solution"), (test_num, target_num))
        
        if not self.is_multilabel and self.is_classification and self.Y.shape[1] > 1:
            self.Y = np.argmax(self.Y, axis=1)
            self.Y_valid = np.argmax(self.Y_valid, axis=1) if self.Y_valid is not None else None
            self.Y_test = np.argmax(self.Y_test, axis=1) if self.Y_test is not None else None
        
    def read_datafile(self, filepath, shape):
        data = []
        with open(filepath, "r") as f:
            for line in f:
                data.append([float(v.strip()) for v in line.split()])
        return np.array(data)

    def read_sparse_datafile(self, filepath, shape):
        data = []
        row_indizes = []
        col_indizes = []
        with open(filepath, "r") as f:
            for row, line in enumerate(f):
                print("\rReading line:",  row, "of", shape[0], end="")
                for value in line.split():
                    value = value.rstrip()

                    data.append(float(value.split(":")[1]))
                    col_indizes.append(int(value.split(":")[0]) - 1)
                    row_indizes.append(row)
            print("Done")
        return csr_matrix((data, (row_indizes, col_indizes)), shape=shape)
    
    def read_binary_sparse_datafile(self, filepath, shape):
        row_indizes = []
        col_indizes = []
        with open(filepath, "r") as f:
            for row, line in enumerate(f):
                print("\rReading line:",  row, "of", shape[0], end="")
                for value in line.split():
                    value = value.rstrip()
                    col_indizes.append(int(value) - 1)
                    row_indizes.append(row)
            print("Done")
        return csr_matrix(([1] * len(row_indizes), (row_indizes, col_indizes)), shape=shape)
