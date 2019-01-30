# -*- encoding: utf-8 -*-
"""
====================
Parallel Usage
====================

*Auto-sklearn* uses *SMAC* to automatically optimize the hyperparameters of
the training models. A variant of *SMAC*, called *pSMAC* (parallel SMAC),
provides a means of running several instances of *auto-sklearn* in a parallel
mode using several computational resources (detailed information of
*pSMAC* can be found `here <https://automl.github.io/SMAC3/stable/psmac.html>`_).
This example shows the necessary steps to configure *auto-sklearn* in
parallel mode.
"""

import multiprocessing
import shutil

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.metrics import accuracy
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
from data_reader import AutoMlReader
from autosklearn.metrics import pac_score

NUM_PROCESSES=4
DATASET = "datasets/automl/tania/tania_public.info"
DATASET_NAME = "tania"
TMP_FOLDER = '/tmp/autosklearn_parallel_example_tmp'
OUTPUT_FOLDER = '/tmp/autosklearn_parallel_example_out'
ENSEMBLE_NBEST = 50
ENSEMBLE_SIZE = 20
TIME_LEFT_FOR_THIS_TASK = 92400
PER_RUN_TIME_LIMIT = 6000
ML_MEMORY_LIMIT = 7000
DELETE_TMP_FOLDER_AFTER_TERMINATE = False
DELETE_OUTPUT_FOLDER_AFTER_TERMINATE = False
RESAMPLING_STRATEGY = 'cv'
RESAMPLING_STRATEGY_ARGUMENTS = {'folds': 5}
INCLUDE_ESTIMATORS = ['DeepFeedNet']
INITIAL_CONFIGURATIONS_VIA_METALEARNING = 0


for dir in [TMP_FOLDER, OUTPUT_FOLDER]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        pass


def get_spawn_classifier(X_train, y_train):
    def spawn_classifier(seed, dataset_name):
        """Spawn a subprocess.

        auto-sklearn does not take care of spawning worker processes. This
        function, which is called several times in the main block is a new
        process which runs one instance of auto-sklearn.
        """

        # Use the initial configurations from meta-learning only in one out of
        # the four processes spawned. This prevents auto-sklearn from evaluating
        # the same configurations in four processes.
        if seed == 0:
            initial_configurations_via_metalearning = INITIAL_CONFIGURATIONS_VIA_METALEARNING
        else:
            initial_configurations_via_metalearning = 0
        smac_scenario_args = {'initial_incumbent': 'RANDOM'} if initial_configurations_via_metalearning > 0 else {}

        # Arguments which are different to other runs of auto-sklearn:
        # 1. all classifiers write to the same output directory
        # 2. shared_mode is set to True, this enables sharing of data between
        # models.
        # 3. all instances of the AutoSklearnClassifier must have a different seed!
        automl = AutoSklearnClassifier(
            time_left_for_this_task=TIME_LEFT_FOR_THIS_TASK, # sec., how long should this seed fit process run
            per_run_time_limit=PER_RUN_TIME_LIMIT, # sec., each model may only take this long before it's killed
            ml_memory_limit=ML_MEMORY_LIMIT, # MB, memory limit imposed on each call to a ML algorithm
            shared_mode=True, # tmp folder will be shared between seeds
            tmp_folder=TMP_FOLDER,
            output_folder=OUTPUT_FOLDER,
            delete_tmp_folder_after_terminate=DELETE_TMP_FOLDER_AFTER_TERMINATE,
            delete_output_folder_after_terminate=DELETE_OUTPUT_FOLDER_AFTER_TERMINATE,
            ensemble_size=0, # ensembles will be built when all optimization runs are finished
            initial_configurations_via_metalearning=initial_configurations_via_metalearning,
            seed=seed,
            smac_scenario_args=smac_scenario_args,
            resampling_strategy=RESAMPLING_STRATEGY,
            resampling_strategy_arguments=RESAMPLING_STRATEGY_ARGUMENTS,
            include_estimators=INCLUDE_ESTIMATORS,
        )
        automl.fit(X_train, y_train, dataset_name=dataset_name)
    return spawn_classifier


def main():

    reader = AutoMlReader(DATASET)
    reader.read()
    X_train, y_train = reader.X, reader.Y

    processes = []
    spawn_classifier = get_spawn_classifier(X_train, y_train)
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(
            target=spawn_classifier,
            args=(i, DATASET_NAME),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print('Starting to build an ensemble!')
    automl = AutoSklearnClassifier(
        time_left_for_this_task=TIME_LEFT_FOR_THIS_TASK,
        per_run_time_limit=PER_RUN_TIME_LIMIT,
        ml_memory_limit=ML_MEMORY_LIMIT,
        shared_mode=True,
        ensemble_size=0,
        tmp_folder=TMP_FOLDER,
        output_folder=OUTPUT_FOLDER,
        delete_tmp_folder_after_terminate=DELETE_TMP_FOLDER_AFTER_TERMINATE,
        delete_output_folder_after_terminate=DELETE_OUTPUT_FOLDER_AFTER_TERMINATE,
        initial_configurations_via_metalearning=0,
        seed=(i + 1),
        resampling_strategy=RESAMPLING_STRATEGY,
        resampling_strategy_arguments=RESAMPLING_STRATEGY_ARGUMENTS,
        include_estimators=INCLUDE_ESTIMATORS,
    )

    # Both the ensemble_size and ensemble_nbest parameters can be changed now if
    # necessary
    automl.fit_ensemble(
        y_train,
        task=MULTILABEL_CLASSIFICATION,
        metric=pac_score,
        precision='32',
        dataset_name=DATASET_NAME,
        ensemble_size=ENSEMBLE_SIZE,
        ensemble_nbest=ENSEMBLE_NBEST,
    )

    # predictions = automl.predict(X_test)
    # print(automl.show_models())
    # print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()