import os
from random import random, randint

import mlflow
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    print("Running mlflow_tracking.py")
    os.environ["MLFLOW_TRACKING_TOKEN"] = "a3ddc03c45ae29fb3376089da11a64a7d26092c1"
    mlflow.set_tracking_uri(
        "http://localhost:30010/projects/zohair/services/pylab-p-zohair-s-ml-flow-1c457/access/5001")
    # mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("my-experiment-2")
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))

    mlflow.log_param("param1", randint(0, 100))

    mlflow.log_metric("foo", random())
    mlflow.log_metric("foo", random() + 1)
    mlflow.log_metric("foo", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    if not os.path.exists("outputs/new_folder"):
        os.makedirs("outputs/new_folder")

    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")

    with open("outputs/test2.txt", "w") as f:
        f.write("hello world!")

    with open("outputs/new_folder/test3.txt", "w") as f:
        f.write("hello world!")

    run = mlflow.active_run()
    mlflow.log_artifacts("outputs", artifact_path="features")

    # List artifacts
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id, "features")
    print(artifacts)
    for artifact in artifacts:
        print(artifact.path)

    # Download artifacts
    client = MlflowClient()
    local_dir = "./downloaded_artifacts"
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)
    local_path = client.download_artifacts(
        run.info.run_id, "features", local_dir)
    print("Artifacts downloaded in: {}".format(local_path))
    print("Artifacts: {}".format(os.listdir(local_path)))
