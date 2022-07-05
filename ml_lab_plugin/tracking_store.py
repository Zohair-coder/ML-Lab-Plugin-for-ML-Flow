from typing import List
from urllib import response
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.entities import (
    Run,
    Experiment,
    ViewType,
    RunInfo,
    RunStatus,
    RunData,
    Param,
    Metric,
    RunTag
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_info import check_run_is_active, check_run_is_deleted
from mlflow.utils.uri import append_to_uri_path
import mlflow.protos.databricks_pb2 as databricks_pb2
from mlflow.exceptions import MlflowException, MissingConfigException
from mlflow.utils.validation import (
    _validate_experiment_id,
    _validate_experiment_name,
    _validate_metric_name,
    _validate_list_experiments_max_results,
    _validate_tag_name,
    _validate_run_id,
    _validate_batch_log_data,
    _validate_batch_log_limits,
    _validate_param_keys_unique
)
from mlflow.store.tracking import SEARCH_MAX_RESULTS_THRESHOLD

from contaxy.clients import JsonDocumentClient
from contaxy.clients.shared import BaseUrlSession
from contaxy.schema.exceptions import ResourceNotFoundError
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR

import json
import uuid
import logging


class MlLabTrackingStore(AbstractStore):
    DEFAULT_EXPERIMENT_ID = "0"

    def __init__(self, store_uri=None, artifact_uri=None):
        print("==============================")
        print("Initialized Tracking Store!")
        print("Store URI: {}".format(store_uri))
        print("Artifact URI: {}".format(artifact_uri))
        print("==============================")
        # TODO: find a solution for not hardcoding the url and token
        self.store_uri = store_uri
        self.artifact_root_uri = artifact_uri
        url = "https://ls6415.wdf.sap.corp:8076/api"
        session = BaseUrlSession(base_url=url)
        token = "32e01d515de63d015e71c2885e968493edbc672f"
        session.headers = {"Authorization": f"Bearer {token}"}
        session.verify = False  # Workaround for development if SAP certificate is not installed
        json_client = JsonDocumentClient(session)
        self.project_id = "2vnuohppfsxosjab4uzqge798"
        self.json_client = json_client

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY, max_results=None, page_token=None):
        from mlflow.utils.search_utils import SearchUtils
        from mlflow.store.entities.paged_list import PagedList
        _validate_list_experiments_max_results(max_results)
        rsl = []
        if view_type == ViewType.ACTIVE_ONLY or view_type == ViewType.ALL:
            rsl += self._get_active_experiments()
        if view_type == ViewType.DELETED_ONLY or view_type == ViewType.ALL:
            rsl += self._get_deleted_experiments()

        experiments = []
        for exp_id in rsl:
            try:
                # trap and warn known issues, will raise unexpected exceptions to caller
                experiment = self._get_experiment(exp_id, view_type)
                if experiment:
                    experiments.append(experiment)
            except MissingConfigException as rnfe:
                # Trap malformed experiments and log warnings.
                logging.warning(
                    "Malformed experiment '%s'. Detailed error %s",
                    str(exp_id),
                    str(rnfe),
                    exc_info=True,
                )
        if max_results is not None:
            experiments, next_page_token = SearchUtils.paginate(
                experiments, page_token, max_results
            )
            return PagedList(experiments, next_page_token)
        else:
            return PagedList(experiments, None)

    def _get_active_experiments(self):
        json_docs = self.json_client.list_json_documents(
            self.project_id, "experiments")

        experiments = []
        for json_doc in json_docs:
            json_value = json.loads(json_doc.json_value)
            if json_value["lifecycle_stage"] == LifecycleStage.ACTIVE:
                exp_id = json_doc.key
                experiments.append(exp_id)
        return experiments

    def _get_deleted_experiments(self):
        json_docs = self.json_client.list_json_documents(
            self.project_id, "experiments")

        experiments = []
        for json_doc in json_docs:
            json_value = json.loads(json_doc.json_value)
            if json_value["lifecycle_stage"] == LifecycleStage.DELETED:
                exp_id = json_doc.key
                experiments.append(exp_id)
        return experiments

    def create_experiment(self, name, artifact_location=None, tags=None):
        print("==============================")
        print("Creating experiment: {}".format(name))
        print("Artifact location: {}".format(artifact_location))
        print("Tags: {}".format(tags))
        print("==============================")
        _validate_experiment_name(name)
        self._validate_experiment_does_not_exist(name)
        # Get all existing experiments and find the one with largest numerical ID.
        # len(list_all(..)) would not work when experiments are deleted.
        experiments_ids = [
            int(e.experiment_id)
            for e in self.list_experiments(ViewType.ALL)
            if e.experiment_id.isdigit()
        ]
        experiment_id = max(experiments_ids) + 1 if experiments_ids else 0
        return self._create_experiment_with_id(name, str(experiment_id), artifact_location, tags)

    def _create_experiment_with_id(self, name, experiment_id, artifact_uri, tags):
        artifact_uri = artifact_uri or append_to_uri_path(
            self.artifact_root_uri, str(experiment_id)
        )
        experiment_dict = {
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_uri,
            "lifecycle_stage": LifecycleStage.ACTIVE,
        }
        self.json_client.create_json_document(
            self.project_id, "experiments", experiment_id, json.dumps(experiment_dict))
        if tags is not None:
            for tag in tags:
                self.set_experiment_tag(experiment_id, tag)
        return experiment_id

    def set_experiment_tag(self, experiment_id, tag):
        _validate_tag_name(tag.key)
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The experiment {} must be in the 'active'"
                "lifecycle_stage to set tags".format(experiment.experiment_id),
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        tag_dict = {"key": tag.key, "value": tag.value,
                    "experiment_id": experiment_id}

        self.json_client.create_json_document(
            self.project_id, "experiment_tags", tag.key, json.dumps(tag_dict))

    def _validate_experiment_does_not_exist(self, name):
        experiment = self.get_experiment_by_name(name)
        if experiment is not None:
            if experiment.lifecycle_stage == LifecycleStage.DELETED:
                raise MlflowException(
                    "Experiment '%s' already exists in deleted state. "
                    "You can restore the experiment, or permanently delete the experiment "
                    "from the .trash folder (under tracking server's root folder) in order to "
                    "use this experiment name again." % experiment.name,
                    databricks_pb2.RESOURCE_ALREADY_EXISTS,
                )
            else:
                raise MlflowException(
                    "Experiment '%s' already exists." % experiment.name,
                    databricks_pb2.RESOURCE_ALREADY_EXISTS,
                )

    def get_experiment(self, experiment_id):
        print("==============================")
        print("Getting experiment: {}".format(experiment_id))
        print("==============================")
        """
        Fetch the experiment.
        Note: This API will search for active as well as deleted experiments.

        :param experiment_id: Integer id for the experiment
        :return: A single Experiment object if it exists, otherwise raises an Exception.
        """
        experiment_id = MlLabTrackingStore.DEFAULT_EXPERIMENT_ID if experiment_id is None else experiment_id
        experiment = self._get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                "Experiment '%s' does not exist." % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        return experiment

    def _get_experiment(self, experiment_id, view_type=ViewType.ALL):
        _validate_experiment_id(experiment_id)
        meta = self._get_experiment_metadata(experiment_id)
        meta["tags"] = self._get_experiment_tags(experiment_id)
        experiment = _read_persisted_experiment_dict(meta)
        if experiment_id != experiment.experiment_id:
            logging.warning(
                "Experiment ID mismatch for exp %s. ID recorded as '%s' in meta data. "
                "Experiment will be ignored.",
                experiment_id,
                experiment.experiment_id,
                exc_info=True,
            )
            return None
        return experiment

    def _get_experiment_metadata(self, experiment_id: str) -> dict:
        response = self.json_client.get_json_document(
            project_id=self.project_id, collection_id="experiments", key=experiment_id)
        return json.loads(response.json_value)

    def _get_experiment_tags(self, experiment_id):
        try:
            response = self.json_client.get_json_document(
                project_id=self.project_id, collection_id="experiments_tags", key=experiment_id)
        except ResourceNotFoundError:
            return dict()

        print("Experiment tags: {}".format(response["json_value"]))
        return response["json_value"]

    def delete_experiment(self, experiment_id):
        print("==============================")
        print("Deleting experiment: {}".format(experiment_id))
        print("==============================")
        json_document = self.json_client.get_json_document(
            self.project_id, "experiments", experiment_id)
        actual_json = json.loads(json_document.json_value)
        if actual_json["lifecycle_stage"] == LifecycleStage.DELETED:
            raise MlflowException(
                "Experiment '%s' already deleted." % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        actual_json["lifecycle_stage"] = LifecycleStage.DELETED
        self.json_client.update_json_document(
            self.project_id, "experiments", experiment_id, json.dumps(actual_json))

    def restore_experiment(self, experiment_id):
        print("==============================")
        print("Restoring experiment: {}".format(experiment_id))
        print("==============================")
        json_document = self.json_client.get_json_document(
            self.project_id, "experiments", experiment_id)
        actual_json = json.loads(json_document.json_value)
        if actual_json["lifecycle_stage"] == LifecycleStage.ACTIVE:
            raise MlflowException(
                "Experiment '%s' already active." % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        actual_json["lifecycle_stage"] = LifecycleStage.ACTIVE
        self.json_client.update_json_document(
            self.project_id, "experiments", experiment_id, json.dumps(actual_json))

    def rename_experiment(self, experiment_id, new_name):
        print("==============================")
        print("Renaming experiment: {}".format(experiment_id))
        print("New name: {}".format(new_name))
        print("==============================")
        json_document = self.json_client.get_json_document(
            self.project_id, "experiments", experiment_id)
        actual_json = json.loads(json_document.json_value)
        if actual_json["lifecycle_stage"] == LifecycleStage.DELETED:
            raise MlflowException(
                "Cannot rename experiment in non-active lifecycle stage."
                " Current stage: %s" % actual_json["lifecycle_stage"]
            )
        actual_json["name"] = new_name
        self.json_client.update_json_document(
            self.project_id, "experiments", experiment_id, json.dumps(actual_json))

    def get_run(self, run_id):
        """
        Note: Will get both active and deleted runs.
        """
        print("==============================")
        print("Getting run: {}".format(run_id))
        print("==============================")
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        if run_info is None:
            raise MlflowException(
                "Run '%s' metadata is in invalid state." % run_id, databricks_pb2.INVALID_STATE
            )
        return self._get_run_from_info(run_info)

    def _get_run_from_info(self, run_info: RunInfo) -> Run:
        metrics = self._get_all_metrics(run_info)
        params = self._get_all_params(run_info)
        tags = self._get_all_tags(run_info)
        return Run(run_info, RunData(metrics, params, tags))

    def _get_all_metrics(self, run_info: RunInfo) -> List[Metric]:
        try:
            response = self.json_client.get_json_document(
                self.project_id, "metrics", run_info.run_uuid)
        except ResourceNotFoundError:
            return []

        response_json = json.loads(response.json_value)
        if type(response_json) == dict:
            response_json = [response_json]
        metrics = []
        for metric in response_json:
            metrics.append(Metric(metric["key"], metric["value"],
                           metric["timestamp"], metric["step"]))
        return metrics

    def _get_all_params(self, run_info: RunInfo) -> List[Param]:
        try:
            response = self.json_client.get_json_document(
                self.project_id, "params", run_info.run_uuid)
        except ResourceNotFoundError:
            return []

        response_json = json.loads(response.json_value)
        if type(response_json) == dict:
            response_json = [response_json]
        params = []
        for param in response_json:
            params.append(Param(param["key"], param["value"]))
        return params

    def _get_all_tags(self, run_info: RunInfo) -> List[RunTag]:
        try:
            response = self.json_client.get_json_document(
                self.project_id, "tags", run_info.run_uuid)
        except ResourceNotFoundError:
            return []

        response_json = json.loads(response.json_value)
        if type(response_json) == dict:
            response_json = [response_json]
        tags = []
        for tag in response_json:
            tags.append(RunTag(tag["key"], tag["value"]))
        return tags

    def _get_run_info(self, run_uuid):
        """
        Note: Will get both active and deleted runs.
        """
        response = self.json_client.get_json_document(
            project_id=self.project_id, collection_id="runs", key=run_uuid)
        return RunInfo.from_dictionary(json.loads(response.json_value))

    def update_run_info(self, run_id, run_status, end_time):
        print("==============================")
        print("Updating run info: {}".format(run_id))
        print("Run status: {}".format(run_status))
        print("End time: {}".format(end_time))
        print("==============================")
        _validate_run_id(run_id)
        json_document = self.json_client.get_json_document(
            self.project_id, "runs", run_id)
        actual_json = json.loads(json_document.json_value)
        actual_json["status"] = RunStatus.to_string(run_status)
        actual_json["end_time"] = end_time
        response = self.json_client.update_json_document(
            self.project_id, "runs", run_id, json.dumps(actual_json))
        response_json = json.loads(response.json_value)
        return RunInfo.from_dictionary(response_json)

    def create_run(self, experiment_id, user_id, start_time, tags=None):
        print("==============================")
        print("Creating run: {}".format(experiment_id))
        print("User ID: {}".format(user_id))
        print("Start time: {}".format(start_time))
        print("Tags: {}".format(tags))
        print("==============================")
        experiment_id = MlLabTrackingStore.DEFAULT_EXPERIMENT_ID if experiment_id is None else experiment_id
        run_uuid = uuid.uuid4().hex
        tags_dict = dict()
        for tag in tags:
            tags_dict[tag.key] = tag.value

        data = {
            "run_uuid": run_uuid,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "status": RunStatus.to_string(RunStatus.RUNNING),
            "start_time": start_time,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "tags": tags_dict
        }
        json_data = json.dumps(data)
        self.json_client.create_json_document(
            project_id=self.project_id, collection_id="runs", key=run_uuid, json_document=json_data)

        return self.get_run(run_uuid)

    def delete_run(self, run_id):
        print("==============================")
        print("Deleting run: {}".format(run_id))
        print("==============================")
        json_document = self.json_client.get_json_document(
            self.project_id, "runs", run_id)
        actual_json = json.loads(json_document.json_value)
        if actual_json["lifecycle_stage"] == LifecycleStage.DELETED:
            raise MlflowException(
                "Run '%s' already deleted." % run_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        actual_json["lifecycle_stage"] = LifecycleStage.DELETED
        self.json_client.update_json_document(
            self.project_id, "runs", run_id, json.dumps(actual_json))

    def restore_run(self, run_id):
        print("==============================")
        print("Restoring run: {}".format(run_id))
        print("==============================")
        json_document = self.json_client.get_json_document(
            self.project_id, "runs", run_id)
        actual_json = json.loads(json_document.json_value)
        if actual_json["lifecycle_stage"] == LifecycleStage.ACTIVE:
            raise MlflowException(
                "Run '%s' already active." % run_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        actual_json["lifecycle_stage"] = LifecycleStage.ACTIVE
        self.json_client.update_json_document(
            self.project_id, "runs", run_id, json.dumps(actual_json))

    def get_metric_history(self, run_id, metric_key):
        print("==============================")
        print("Getting metric history: {}".format(run_id))
        print("Metric key: {}".format(metric_key))
        print("==============================")
        _validate_run_id(run_id)
        _validate_metric_name(metric_key)
        # TODO: Implement this
        return None

    def _search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):
        print("==============================")
        print("Searching runs")
        print("Experiment IDs: {}".format(experiment_ids))
        print("Filter string: {}".format(filter_string))
        print("Run view type: {}".format(run_view_type))
        print("Max results: {}".format(max_results))
        print("Order by: {}".format(order_by))
        print("Page token: {}".format(page_token))
        print("==============================")

        from mlflow.utils.search_utils import SearchUtils

        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at "
                "most {}, but got value {}".format(
                    SEARCH_MAX_RESULTS_THRESHOLD, max_results),
                databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        runs = []
        for experiment_id in experiment_ids:
            run_infos = self._list_run_infos(experiment_id, run_view_type)
            runs.extend(self._get_run_from_info(r) for r in run_infos)
        filtered = SearchUtils.filter(runs, filter_string)
        sorted_runs = SearchUtils.sort(filtered, order_by)
        runs, next_page_token = SearchUtils.paginate(
            sorted_runs, page_token, max_results)
        return runs, next_page_token

    def _list_run_infos(self, experiment_id, view_type):
        response = self.json_client.list_json_documents(
            self.project_id, "runs")
        runs = []
        for json_document in response:
            run_info = RunInfo.from_dictionary(
                json.loads(json_document.json_value))
            if run_info.experiment_id == experiment_id and LifecycleStage.matches_view_type(view_type, run_info.lifecycle_stage):
                runs.append(run_info)
        return runs

    def log_batch(self, run_id, metrics, params, tags):
        print("==============================")
        print("Logging batch")
        print("==============================")
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        try:
            for param in params:
                self._log_run_param(run_info, param)
            for metric in metrics:
                self._log_run_metric(run_info, metric)
            for tag in tags:
                self._set_run_tag(run_info, tag)
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)
        return None

    def _log_run_param(self, run_info: RunInfo, param: Param) -> None:
        data = {
            "key": param.key,
            "value": param.value,
            "run_uuid": run_info.run_uuid
        }
        self.json_client.create_json_document(
            self.project_id, "params", run_info.run_uuid, json.dumps(data))

    def _log_run_metric(self, run_info: RunInfo, metric: Metric) -> None:
        data = {
            "key": metric.key,
            "value": metric.value,
            "timestamp": metric.timestamp,
            "run_uuid": run_info.run_uuid,
            "step": metric.step,
        }
        self.json_client.create_json_document(
            self.project_id, "metrics", run_info.run_uuid, json.dumps(data))

    def _set_run_tag(self, run_info: RunInfo, tag: RunTag) -> None:
        data = {
            "key": tag.key,
            "value": tag.value,
            "run_uuid": run_info.run_uuid
        }
        self.json_client.create_json_document(
            self.project_id, "tags", run_info.run_uuid, json.dumps(data))


def _read_persisted_experiment_dict(experiment_dict: dict) -> Experiment:
    dict_copy = experiment_dict.copy()

    # 'experiment_id' was changed from int to string, so we must cast to string
    # when reading legacy experiments
    if isinstance(dict_copy["experiment_id"], int):
        dict_copy["experiment_id"] = str(dict_copy["experiment_id"])
    return Experiment.from_dictionary(dict_copy)
