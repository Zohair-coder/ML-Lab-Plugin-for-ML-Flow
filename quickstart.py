import os
from random import random, randint

import mlflow
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    print("Running mlflow_tracking.py")
    mlflow.set_tracking_uri("http://localhost:5001")
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))

    # mlflow.log_param("param1", randint(0, 100))

    # mlflow.log_metric("foo", random())
    # mlflow.log_metric("foo", random() + 1)
    # mlflow.log_metric("foo", random() + 2)

    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    with open("outputs/test2.txt", "w") as f:
        f.write("hello world!")

    with open("outputs/new_folder/test3.txt", "w") as f:
        f.write("hello world!")

    with mlflow.start_run() as run:
        mlflow.log_artifacts("outputs", artifact_path="features")

    # client = MlflowClient()
    # artifacts = client.list_artifacts(run.info.run_id, "features")
    # print("Artifacts: {}".format(artifacts))

    # Download artifacts
    client = MlflowClient()
    local_dir = "./downloaded_artifacts"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    local_path = client.download_artifacts(run.info.run_id, "features", local_dir)
    print("Artifacts downloaded in: {}".format(local_path))
    print("Artifacts: {}".format(os.listdir(local_path)))