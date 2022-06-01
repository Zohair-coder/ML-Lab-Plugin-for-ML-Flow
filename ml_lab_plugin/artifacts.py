from mlflow.store.artifact.artifact_repo import ArtifactRepository


class MlLabArtifactRepository(ArtifactRepository):
    is_plugin = True
