from data_management.data_manager import DataManager, deterministic_shuffle_and_split
import os

def load_dataset(dataset_index, test_split=0.2, valid_split=0.2, random_seed=0):
    dataset_name = None
    with open(os.path.join(os.path.dirname(__file__), "datasets.txt"), "r") as f:
        for i, line in enumerate(f):
            if i == dataset_index:
                dataset_name = line.strip()
    
    if dataset_name is not None:
        dm = DataManager(verbose=True)
        dm.read_data(dataset_name, test_split=0.2, is_classification=True, random_seed=random_seed)
        X_test = dm.X_test
        Y_test = dm.Y_test

        _, _, X_train, Y_train, X_valid, Y_valid = deterministic_shuffle_and_split(dm.X_train, dm.Y_train, split=valid_split, seed=random_seed)
        return {
            "X_train": X_train,
            "Y_train": Y_train,
            "X_valid": X_valid,
            "Y_valid": Y_valid,
            "X_test": X_test,
            "Y_test": Y_test
        }