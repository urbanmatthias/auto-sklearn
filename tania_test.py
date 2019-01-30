from data_reader import AutoMlReader
import autosklearn.classification

reader = AutoMlReader("/home/matthias/Dokumente/Uni/Auto-PyTorch/datasets/automl/tania/tania_public.info")
reader.read()
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=500,
    per_run_time_limit=500,
    tmp_folder='/tmp/autosklearn_cv_example_tmp',
    output_folder='/tmp/autosklearn_cv_example_out',
    delete_tmp_folder_after_terminate=False,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},
    include_estimators=['DeepFeedNet'])
automl.fit(reader.X, reader.Y, dataset_name='tania')

# automl.refit(X_train.copy(), y_train.copy())
# print(automl.show_models())

# predictions = automl.predict(X_test)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
