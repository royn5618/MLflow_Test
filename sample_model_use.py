import mlflow
import pandas as pd


def get_last_run_model():
    """

    :return:
    """
    last_run_info = mlflow.mlflow.list_run_infos('0')[0]
    return last_run_info.run_id


def load_model(logged_model):
    """

    :param logged_model:
    :return:
    """
    return mlflow.pyfunc.load_model(logged_model)


if __name__ == "__main__":
    model_run_id = get_last_run_model()
    logged_model = 'runs:/' + model_run_id + '/model'
    loaded_model = load_model(logged_model)
    sample_data = [[8.1, 0.28, 0.4, 6.9, 0.05, 30, 97, 0.9951, 3.26, 0.44, 10.1]]
    prediction = loaded_model.predict(pd.DataFrame(sample_data))
    print(prediction)
