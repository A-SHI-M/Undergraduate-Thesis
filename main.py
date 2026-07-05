import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from Anomaly_Detection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from Anomaly_Detection.pipeline.stage_03_model_training import TrainingPipeline
from Anomaly_Detection.utils.visualizations import Visualizer
from Anomaly_Detection.config.configuaration import ConfigurationManager


def run_dataset(dataset_name: str):
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    # Stage 1 — Data preparation
    logger.info(">>>>>> Stage 1: Data Transformation <<<<<<")
    DataIngestionPipeline(dataset_name=dataset_name).run()
    logger.info(">>>>>> Stage 1 complete <<<<<<")

    # Stage 2 — Train & evaluate all models
    logger.info(">>>>>> Stage 2: Training all models <<<<<<")
    pipeline = TrainingPipeline(dataset_name=dataset_name)
    cfg = pipeline.cfg
    eval_cfg = cfg.get_model_evaluation_config()
    viz = Visualizer()

    _, x_test, y_test = pipeline._load_data()
    model_store = {}
    results = {}

    logger.info("Training FC-AE")
    results["FC-AE"] = pipeline.run_fc_ae()

    logger.info("Training CNN-AE")
    results["CNN-AE"] = pipeline.run_cnn_ae()

    logger.info("Training VAE")
    results["VAE"] = pipeline.run_vae()

    logger.info("Training Beta-VAE")
    results["Beta-VAE"] = pipeline.run_beta_vae()

    logger.info("Training CVAE")
    results["CVAE"] = pipeline.run_cvae()

    logger.info("Training VQ-VAE")
    results["VQ-VAE"] = pipeline.run_vqvae()

    logger.info("Training Normal BiGAN")
    results["BiGAN"] = pipeline.run_normal_bigan()

    logger.info("Training GANomaly")
    results["GANomaly"] = pipeline.run_ganomaly()

    logger.info("Training f-AnoGAN")
    results["f-AnoGAN"] = pipeline.run_anogan()

    logger.info("Training Improved BiGAN")
    results["Improved-BiGAN"] = pipeline.run_bigan(variant="full")

    logger.info(">>>>>> Stage 2 complete <<<<<<")

    # Stage 3 — Overall comparison plots
    logger.info(">>>>>> Stage 3: Comparison plots <<<<<<")
    comp_dir = Path(eval_cfg.comparison_dir)
    comp_dir.mkdir(parents=True, exist_ok=True)
    viz.plot_comparison(results, comp_dir)
    viz.plot_roc_comparison(results, comp_dir)
    logger.info(">>>>>> Stage 3 complete <<<<<<")

    # Stage 4 — T-SNE latent space
    logger.info(">>>>>> Stage 4: T-SNE latent space <<<<<<")
    pipeline.run_tsne(model_store, x_test, y_test)
    logger.info(">>>>>> Stage 4 complete <<<<<<")

    # Stage 5 — Ablation study
    logger.info(">>>>>> Stage 5: Ablation study <<<<<<")
    pipeline.run_ablation()
    logger.info(">>>>>> Stage 5 complete <<<<<<")

    logger.info(f"Results saved to resultAnalysis/{cfg.display_name}/ "
                f"and ablationAnalysis/{cfg.display_name}/")


def main():
    # Discover all configured datasets from config.yaml
    root_cfg = ConfigurationManager()
    all_datasets = list(root_cfg.config["data_ingestion"]["datasets"].keys())
    logger.info(f"Configured datasets: {all_datasets}")

    for dataset_name in all_datasets:
        run_dataset(dataset_name)

    logger.info("All datasets complete.")


if __name__ == "__main__":
    main()
