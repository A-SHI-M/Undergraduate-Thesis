from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_transformation import DataTransformation


class DataIngestionPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name).get_data_ingestion_config()

    def run(self):
        DataTransformation(self.cfg).transform()


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run()
