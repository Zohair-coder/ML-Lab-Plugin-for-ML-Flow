from mlflow.store.tracking.abstract_store import AbstractStore


class MlLabTrackingStore(AbstractStore):

    def __init__(self, store_uri=None, artifact_uri=None):
        self.is_plugin = True
        print("==============================")
        print("Initialized!")
        print("==============================")
    
    def list_experiments(self):
        print("==============================")
        print("Listing experiments")
        print("==============================")
        return []
    
    def create_experiment(self, name, artifact_location=None, tags=None):
        print("==============================")
        print("Creating experiment: {}".format(name))
        print("Artifact location: {}".format(artifact_location))
        print("Tags: {}".format(tags))
        print("==============================")
        return None
    
    def get_experiment(self, experiment_id):
        print("==============================")
        print("Getting experiment: {}".format(experiment_id))
        print("==============================")
        return None
    
    def delete_experiment(self, experiment_id):
        print("==============================")
        print("Deleting experiment: {}".format(experiment_id))
        print("==============================")
        return None
    
    def restore_experiment(self, experiment_id):
        print("==============================")
        print("Restoring experiment: {}".format(experiment_id))
        print("==============================")
        return None
    
    def rename_experiment(self, experiment_id, new_name):
        print("==============================")
        print("Renaming experiment: {}".format(experiment_id))
        print("New name: {}".format(new_name))
        print("==============================")
        return None
    
    def get_run(self, run_id):
        print("==============================")
        print("Getting run: {}".format(run_id))
        print("==============================")
        return None
    
    def update_run_info(self, run_id, run_status, end_time):
        print("==============================")
        print("Updating run info: {}".format(run_id))
        print("Run status: {}".format(run_status))
        print("End time: {}".format(end_time))
        print("==============================")
        return None

    def create_run(self, experiment_id, user_id, start_time, tags=None):
        print("==============================")
        print("Creating run: {}".format(experiment_id))
        print("User ID: {}".format(user_id))
        print("Start time: {}".format(start_time))
        print("Tags: {}".format(tags))
        print("==============================")
        return None
    
    def delete_run(self, run_id):
        print("==============================")
        print("Deleting run: {}".format(run_id))
        print("==============================")
        return None
    
    def restore_run(self, run_id):
        print("==============================")
        print("Restoring run: {}".format(run_id))
        print("==============================")
        return None
    
    def get_metric_history(self, run_id, metric_key):
        print("==============================")
        print("Getting metric history: {}".format(run_id))
        print("Metric key: {}".format(metric_key))
        print("==============================")
        return None
    
    def _search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token):
        print("==============================")
        print("Searching runs")
        print("==============================")
        return None
    
    def log_batch(self, run_id, metrics, params, tags, artifact_bodies):
        print("==============================")
        print("Logging batch")
        print("==============================")
        return None
    
    def record_logged_model(self, run_id, model_uri):
        print("==============================")
        print("Recording logged model")
        print("==============================")
        return None
    

