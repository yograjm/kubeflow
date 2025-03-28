from kfp.dsl import component, pipeline, Input, Output, Dataset, Model
import kfp

# Pipeline Component-1
# Applying `component` decorator on the `create_dataset` function
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def create_dataset(drop_missing_vals: bool, iris_dataset: Output[Dataset]):
    import pandas as pd
    from sklearn import datasets

    # Load dataset
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    if drop_missing_vals:
        df = df.dropna()

    with open(iris_dataset.path, 'w') as file:
        df.to_csv(file, index=False)


# Pipeline Component-2
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_test_splitting(
    input_iris_dataset: Input[Dataset], 
    X_train: Output[Dataset], 
    X_test: Output[Dataset], 
    y_train: Output[Dataset], 
    y_test: Output[Dataset],
    test_size: float,
    random_state: int,
):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    with open(input_iris_dataset.path) as file:
        df = pd.read_csv(file)

    X = df.drop(['species'], axis=1)
    y = df[['species']]

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    X_train_data.to_csv(X_train.path, index=False)
    X_test_data.to_csv(X_test.path, index=False)
    y_train_data.to_csv(y_train.path, index=False)
    y_test_data.to_csv(y_test.path, index=False)


# Pipeline Component-3
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def training_classifier_model(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    save_model: Output[Model],
    n_estimators: int,
    random_state: int,
):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    X_train_data = pd.read_csv(X_train.path)
    y_train_data = pd.read_csv(y_train.path)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_data, y_train_data['species'].values)

    with open(save_model.path, 'wb') as file:
        pickle.dump(model, file)


# Pipeline Component-4
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def check_model_performance(
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model_path: Input[Model],
    metrics_data: Output[Dataset]
):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    X_test_data = pd.read_csv(X_test.path)
    y_test_data = pd.read_csv(y_test.path)

    model = joblib.load(model_path.path)
    y_pred = model.predict(X_test_data)

    acc = accuracy_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred, average='weighted')
    prec = precision_score(y_test_data, y_pred, average='weighted')
    rec = recall_score(y_test_data, y_pred, average='weighted')

    metrics_df = pd.DataFrame({'Accuracy': [acc],
                               'F1-score': [f1],
                               'Precision': [prec],
                               'Recall': [rec]})
    
    with open(metrics_data.path, 'w') as file:
        metrics_df.to_csv(file, index=False)


# Pipeline Component-5
# @component(
#     packages_to_install=["pandas", "numpy", "scikit-learn"],
#     base_image="python:3.10-slim",
# )
# def inference_with_model(
#     model_path: Input[Model],
#     predictions: Output[Dataset]
# ):
#     import numpy as np
#     import pandas as pd
#     import joblib
    
#     model = joblib.load(model_path.path)

#     input_df = pd.DataFrame({'sepal length (cm)': [2.5],
#                              'sepal width (cm)': [3.5],
#                              'petal length (cm)': [4.5],
#                              'petal width (cm)': [5.5]})

#     pred = model.predict(input_df)
#     target_names = ['setosa', 'versicolor', 'virginica']
#     label = target_names[pred[0]]

#     pred_df = input_df.copy()
#     pred_df['species'] = label

#     with open(predictions.path, 'w') as file:
#         pred_df.to_csv(file, index=False)
    

# Pipeline

@pipeline(
    name="iris-training-pipeline",
    description="An ML pipeline for data preparation, train/test splitting, model training, performance evaluation, and inference."
)
def ml_pipeline(
    drop_missing_vals: bool = True,
    test_size: float = 0.25,
    random_state: int = 42,
    n_estimators: int = 100,
):
    create_dataset_task = create_dataset(drop_missing_vals=drop_missing_vals)

    train_test_split_task = train_test_splitting(
        input_iris_dataset=create_dataset_task.outputs['iris_dataset'],
        test_size=test_size,
        random_state=random_state)
    
    training_task = training_classifier_model(
        X_train=train_test_split_task.outputs['X_train'],
        y_train=train_test_split_task.outputs['y_train'],
        n_estimators=n_estimators,
        random_state=random_state)
    
    performance_task = check_model_performance(
        X_test=train_test_split_task.outputs['X_test'],
        y_test=train_test_split_task.outputs['y_test'],
        model_path=training_task.outputs['save_model'])

    # inference_task = inference_with_model(model_path=training_task.outputs['save_model'])



# Compile the pipeline to an IR YAML file, then upload it to Kubeflow UI to create pipeline
if __name__ == '__main__':
    compiler = kfp.compiler.Compiler()
    compiler.compile(ml_pipeline, "ml_pipeline_v1.yaml")
    print("YAML file created successfully!")
