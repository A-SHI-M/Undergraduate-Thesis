import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from Anomaly_Detection.pipeline.training_pipeline import TrainingPipeline

MODELS = ["fc_ae", "cnn_ae", "vae", "beta_vae", "cvae", "vqvae", "bigan", "all"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="all",
        help=(
            "Model to train:\n"
            "  fc_ae    — Fully Connected Autoencoder\n"
            "  cnn_ae   — Convolutional Autoencoder\n"
            "  vae      — Variational Autoencoder\n"
            "  beta_vae — Beta-VAE\n"
            "  cvae     — Conditional VAE\n"
            "  vqvae    — VQ-VAE\n"
            "  bigan    — Improved BiGAN\n"
            "  all      — Train every model in sequence  (default)"
        ),
    )
    return parser.parse_args()


def run(stage: str, fn):
    try:
        logger.info(f">>>>>> {stage} started <<<<<<")
        fn()
        logger.info(f">>>>>> {stage} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrainingPipeline()

    dispatch = {
        "fc_ae":    lambda: run("FC Autoencoder",   pipeline.run_fc_ae),
        "cnn_ae":   lambda: run("CNN Autoencoder",  pipeline.run_cnn_ae),
        "vae":      lambda: run("VAE",              pipeline.run_vae),
        "beta_vae": lambda: run("Beta-VAE",         pipeline.run_beta_vae),
        "cvae":     lambda: run("Conditional VAE",  pipeline.run_cvae),
        "vqvae":    lambda: run("VQ-VAE",           pipeline.run_vqvae),
        "bigan":    lambda: run("Improved BiGAN",   pipeline.run_bigan),
    }

    if args.model == "all":
        for fn in dispatch.values():
            fn()
    else:
        dispatch[args.model]()
