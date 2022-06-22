from mlflow.store.artifact.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.entities import FileInfo
from mlflow.utils.file_utils import relative_path_to_artifact_path
from contaxy.clients import FileClient
from contaxy.clients.shared import BaseUrlSession

import os
import posixpath


class MlLabArtifactRepository(ArtifactRepository):
    is_plugin = True

    def __init__(self, artifact_uri):
        # TODO: find a solution for not hardcoding the url and token
        url = "https://ls6415.wdf.sap.corp:8076/api"
        session = BaseUrlSession(base_url=url)
        token = "32e01d515de63d015e71c2885e968493edbc672f"
        session.headers = {"Authorization": f"Bearer {token}"}
        session.verify = False  # Workaround for development if SAP certificate is not installed
        file_client = FileClient(session)
        self.project_id = "2vnuohppfsxosjab4uzqge798"
        self.file_client = file_client

    def log_artifact(self, local_file, artifact_path=None):
        print("==============================")
        print("Logging artifact: {}".format(local_file))
        print("Artifact path: {}".format(artifact_path))
        print("==============================")
        verify_artifact_path(artifact_path)
        file_name = os.path.basename(local_file)
        artifact_path = os.path.join(artifact_path, file_name)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)

        self.file_client.upload_file(
            project_id=self.project_id, file_key=artifact_path, file_stream=open(local_file, "rb"))

    def log_artifacts(self, local_dir, artifact_path=None):
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            if root == local_dir:
                artifact_dir = artifact_path
            else:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_dir = (
                    posixpath.join(
                        artifact_path, rel_path) if artifact_path else rel_path
                )
            for f in filenames:
                self.log_artifact(os.path.join(root, f), artifact_dir)

    def list_artifacts(self, path):
        print("==============================")
        print("Listing artifacts: {}".format(path))
        print("==============================")
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if path:
            path = os.path.normpath(path)
        files = self.file_client.list_files(
            project_id=self.project_id, prefix=path)
        infos = []
        for file in files:
            # remove the leading slash as well
            file_path = file.key[len(path)+1:]
            print(file_path)
            if "/" in file_path:
                folder_name = file_path.split("/")[0]
                info = FileInfo(folder_name, is_dir=True, file_size=None)
            else:
                info = FileInfo(file.display_name, False, file.file_size)
            infos.append(info)
        return infos

    def _download_file(self, remote_file_path, local_path):
        print("==============================")
        print("Downloading file: {}".format(remote_file_path))
        print("Local path: {}".format(local_path))
        print("==============================")
        stream = self.file_client.download_file(
            project_id=self.project_id, file_key=remote_file_path)
        with open(local_path, "wb") as f:
            for chunk in stream:
                f.write(chunk)
