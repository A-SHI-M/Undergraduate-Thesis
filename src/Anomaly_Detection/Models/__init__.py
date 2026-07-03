from .VAE import VAELossLayer, sampling as vae_sampling, build_vae
from .Conditional_VAE import CVAELossLayer, sampling as cvae_sampling, build_cvae
from .Beta_VAE import build_beta_vae
from .FullyConnectedAutoencoder import build_fc_autoencoder
from .ConvolutionalAutoencoder import build_cnn_autoencoder
from .VQ_VAE import VectorQuantizer, build_vqvae
from .ImprovedBIGAN import (
    build_encoder,
    build_generator,
    build_discriminator,
    build_bigan,
    build_reconstruction_model,
    ssim_loss,
)
from .GANomaly import (
    build_ganomaly_encoder,
    build_ganomaly_generator,
    build_ganomaly_discriminator,
    build_ganomaly_model,
)
from .fAnoGAN import (
    build_fanogan_generator,
    build_fanogan_discriminator,
    build_fanogan_dcgan,
    build_fanogan_encoder,
)

__all__ = [
    # VAE
    "VAELossLayer",
    "vae_sampling",
    "build_vae",
    # Beta-VAE
    "build_beta_vae",
    # CVAE
    "CVAELossLayer",
    "cvae_sampling",
    "build_cvae",
    # Autoencoders
    "build_fc_autoencoder",
    "build_cnn_autoencoder",
    # VQ-VAE
    "VectorQuantizer",
    "build_vqvae",
    # Improved BiGAN
    "build_encoder",
    "build_generator",
    "build_discriminator",
    "build_bigan",
    "build_reconstruction_model",
    "ssim_loss",
    # GANomaly
    "build_ganomaly_encoder",
    "build_ganomaly_generator",
    "build_ganomaly_discriminator",
    "build_ganomaly_model",
    # f-AnoGAN
    "build_fanogan_generator",
    "build_fanogan_discriminator",
    "build_fanogan_dcgan",
    "build_fanogan_encoder",
]
