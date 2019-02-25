# -*- encoding: utf-8 -*-

import sys
import os
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from data_management.data_manager import DataManager, deterministic_shuffle_and_split
import argparse
from autosklearn.metrics import balanced_accuracy
import tempfile

import autosklearn.classification


def main(dataset_name, dataset_id):
    dm = DataManager()
    dm.read_data(dataset_name)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=92400, # sec., how long should this seed fit process run
        per_run_time_limit=6000, # sec., each model may only take this long before it's killed
        ml_memory_limit=8000, # MB, memory limit imposed on each call to a ML algorithm
        tmp_folder=os.path.join(tempfile.gettempdir(), "tmp"),
        output_folder=os.path.join(tempfile.gettempdir(), "output"),
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        ensemble_size=50,
        initial_configurations_via_metalearning=0,
        seed=0,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        include_estimators=['DeepFeedNet']
    )

    _ , _, X_train, y_train, X_test, y_test = deterministic_shuffle_and_split(dm.X, dm.Y, 0.2, 0)
    automl.fit(X_train, y_train, X_test=X_test, y_test=y_test, dataset_name=dataset_name, metric=balanced_accuracy)

if __name__ == '__main__':
    dataset_id = int(sys.argv[-1]) - 1

    print(tempfile.gettempdir())
    os.environ['THEANO_FLAGS'] = "base_compiledir=%s" % os.path.join(tempfile.gettempdir(), "compile")
    os.environ['OPENBLAS_NUM_THREADS'] = "1"

    with open("openml_datasets.txt", "r") as f:
        dataset_name = list(f)[dataset_id].strip()
    main(dataset_name, dataset_id)
