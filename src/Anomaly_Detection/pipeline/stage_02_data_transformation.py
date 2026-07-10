import sys
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_transformation import DataTransformation
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException


class DataTransformationPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name).get_data_transformation_config()

    def run(self):
        try:
            logger.info("Starting data transformation")
            DataTransformation(self.cfg).initiate()
            logger.info("Data transformation complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


if __name__ == "__main__":
    root_cfg = ConfigurationManager()
    all_datasets = list(root_cfg.config["data_ingestion"]["datasets"].keys())
    logger.info(f"Datasets to transform: {all_datasets}")
    for ds in all_datasets:
        logger.info(f"--- {ds} ---")
        DataTransformationPipeline(dataset_name=ds).run()
