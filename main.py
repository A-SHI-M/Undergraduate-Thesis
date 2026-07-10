import sys
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from Anomaly_Detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from Anomaly_Detection.pipeline.stage_03_model_training import TrainingPipeline
from Anomaly_Detection.pipeline.stage_04_model_evaluation import EvaluationPipeline


def run_dataset(dataset_name: str):
    try:
        logger.info(f"{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        logger.info(">>>>>> Stage 1: Data Ingestion <<<<<<")
        DataIngestionPipeline(dataset_name=dataset_name).run()
        logger.info(">>>>>> Stage 1 complete <<<<<<\n")

        logger.info(">>>>>> Stage 2: Data Transformation <<<<<<")
        DataTransformationPipeline(dataset_name=dataset_name).run()
        logger.info(">>>>>> Stage 2 complete <<<<<<\n")

        logger.info(">>>>>> Stage 3: Model Training <<<<<<")
        training = TrainingPipeline(dataset_name=dataset_name)
        training.run_fc_ae()
        training.run_cnn_ae()
        training.run_vae()
        training.run_beta_vae()
        training.run_cvae()
        training.run_vqvae()
        training.run_ganomaly()
        training.run_fanogan()
        training.run_normal_bigan()
        training.run_bigan(variant="full")
        training.run_ablation()
        logger.info(">>>>>> Stage 3 complete <<<<<<\n")

        logger.info(">>>>>> Stage 4: Model Evaluation <<<<<<")
        EvaluationPipeline(dataset_name=dataset_name).run()
        logger.info(">>>>>> Stage 4 complete <<<<<<\n")

    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e


def main():
    try:
        root_cfg = ConfigurationManager()
        all_datasets = list(root_cfg.config["data_ingestion"]["datasets"].keys())
        logger.info(f"Configured datasets: {all_datasets}")

        for dataset_name in all_datasets:
            run_dataset(dataset_name)

        logger.info("All datasets complete.")
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e


if __name__ == "__main__":
    main()
