# 4. Beta-VAE
# Parameters
latent_dim_beta = 128
beta_value = 10.0

# encoder
inputs_beta = Input(shape=(input_dim,))
x = Dense(1024, activation='relu')(inputs_beta)
x = Dense(512, activation='relu')(x)
z_mean_beta = Dense(latent_dim_beta)(x)
z_log_var_beta = Dense(latent_dim_beta)(x)
z_beta = Lambda(sampling)([z_mean_beta, z_log_var_beta])

# decoder
x = Dense(512, activation='relu')(z_beta)
x = Dense(1024, activation='relu')(x)
outputs_beta = Dense(input_dim, activation='sigmoid')(x)

beta_vae_output = VAELossLayer(beta=beta_value)([inputs_beta, outputs_beta, z_mean_beta, z_log_var_beta])

# Beta-VAE model
beta_vae = Model(inputs_beta, beta_vae_output)
beta_vae.compile(optimizer='adam')