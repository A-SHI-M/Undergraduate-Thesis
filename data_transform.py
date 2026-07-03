import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from Anomaly_Detection.pipeline.data_transformation_pipeline import DataIngestionPipeline


if __name__ == "__main__":
    try:
        logger.info(">>>>>> Data Transformation started <<<<<<")
        DataIngestionPipeline().run()
        logger.info(">>>>>> Data Transformation completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise
