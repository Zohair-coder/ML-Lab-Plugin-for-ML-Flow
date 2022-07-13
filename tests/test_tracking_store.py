import pytest
import mlflow
import os
import shutil
import uuid
from helpers import get_safe_port, launch_tracking_store_test_server
from mlflow.tracking import MlflowClient
from mlflow.entities import Run


@pytest.fixture(scope="module", autouse=True)
def artifacts_server():
    """
    Starts and stops mlflow server and sets the tracking_uri.
    """
    store_uri = "ml-lab:/mlflow"
    # store_uri = "./mlruns"
    port = get_safe_port()
    process = launch_tracking_store_test_server(store_uri, port)
    mlflow.set_tracking_uri("http://localhost:{}".format(port))
    yield
    process.kill()
    if os.path.isdir("mlruns"):
        shutil.rmtree("mlruns")


@pytest.fixture(scope="function")
def run(artifacts_server):
    """
    Creates a run and returns the run object.
    """
    mlflow.start_run()
    yield mlflow.active_run()
    mlflow.end_run()


@pytest.fixture(scope="module")
def client() -> MlflowClient:
    """"
    Returns an instance of MlflowClient.
    """
    return MlflowClient()


@pytest.fixture(scope="module")
def experiment(client: MlflowClient) -> str:
    """"
    Returns an Mlflow Experiment.
    """
    experiment_name = str(uuid.uuid4())
    experiment_id = client.create_experiment(experiment_name, "./mlruns")
    yield experiment_id
    if experiment in [e.experiment_id for e in client.list_experiments()]:
        client.delete_experiment(experiment_id)


def test_zero_metrics(client: MlflowClient, run: Run) -> None:
    assert len(client.get_run(run.info.run_id).data.metrics) == 0


def test_zero_params(client: MlflowClient, run: Run) -> None:
    assert len(client.get_run(run.info.run_id).data.params) == 0


def test_default_tags(client: MlflowClient, run: Run) -> None:
    tags = client.get_run(run.info.run_id).data.tags
    # might have mlflow.source.git.commit
    assert len(tags) == 3 or len(tags) == 4
    assert tags["mlflow.user"]
    assert tags["mlflow.source.name"]
    assert tags["mlflow.source.type"]


def test_log_one_metric(client: MlflowClient, run: Run) -> None:
    mlflow.log_metric("metric", 5)
    assert len(client.get_run(run.info.run_id).data.metrics) == 1
    assert client.get_run(run.info.run_id).data.metrics["metric"] == 5


def test_log_one_param(client: MlflowClient, run: Run) -> None:
    mlflow.log_param("param", "value")
    assert len(client.get_run(run.info.run_id).data.params) == 1
    assert client.get_run(run.info.run_id).data.params["param"] == "value"


def test_log_one_tag(client: MlflowClient, run: Run) -> None:
    previous_tags = client.get_run(run.info.run_id).data.tags
    mlflow.set_tag("tag", "value")
    new_tags = client.get_run(run.info.run_id).data.tags
    assert len(new_tags) == len(previous_tags) + 1
    assert new_tags["tag"] == "value"


def test_log_multiple_metrics(client: MlflowClient, run: Run) -> None:
    mlflow.log_metric("metric1", 5)
    mlflow.log_metric("metric2", 10)
    assert len(client.get_run(run.info.run_id).data.metrics) == 2
    assert client.get_run(run.info.run_id).data.metrics["metric1"] == 5
    assert client.get_run(run.info.run_id).data.metrics["metric2"] == 10


def test_log_multiple_params(client: MlflowClient, run: Run) -> None:
    mlflow.log_param("param1", "value1")
    mlflow.log_param("param2", "value2")
    assert len(client.get_run(run.info.run_id).data.params) == 2
    assert client.get_run(run.info.run_id).data.params["param1"] == "value1"
    assert client.get_run(run.info.run_id).data.params["param2"] == "value2"


def test_log_multiple_tags(client: MlflowClient, run: Run) -> None:
    previous_tags = client.get_run(run.info.run_id).data.tags
    mlflow.set_tag("tag1", "value1")
    mlflow.set_tag("tag2", "value2")
    new_tags = client.get_run(run.info.run_id).data.tags
    assert len(new_tags) == len(previous_tags) + 2
    assert new_tags["tag1"] == "value1"
    assert new_tags["tag2"] == "value2"


def test_log_multiple_tags_with_same_key(client: MlflowClient, run: Run) -> None:
    previous_tags = client.get_run(run.info.run_id).data.tags
    mlflow.set_tag("tag", "value1")
    mlflow.set_tag("tag", "value2")
    new_tags = client.get_run(run.info.run_id).data.tags
    assert len(new_tags) == len(previous_tags) + 1
    assert new_tags["tag"] == "value2"


def test_log_non_numeric_metric() -> None:
    with pytest.raises(TypeError):
        mlflow.log_metric("metric", "string")


def test_list_new_experiment(client: MlflowClient, experiment: str) -> None:
    print("Experiments: {}".format(client.list_experiments()))
    assert experiment in [e.experiment_id for e in client.list_experiments()]
