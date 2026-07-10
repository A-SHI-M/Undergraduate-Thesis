import sys
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_ingestion import DataIngestion
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException


class DataIngestionPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name).get_data_ingestion_config()

    def run(self):
        try:
            logger.info("Starting data ingestion")
            DataIngestion(self.cfg).download()
            logger.info("Data ingestion complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


if __name__ == "__main__":
    root_cfg = ConfigurationManager()
    all_datasets = list(root_cfg.config["data_ingestion"]["datasets"].keys())
    logger.info(f"Datasets to download: {all_datasets}")
    for ds in all_datasets:
        logger.info(f"--- {ds} ---")
        DataIngestionPipeline(dataset_name=ds).run()
