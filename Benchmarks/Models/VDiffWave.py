import tensorflow as tf
import numpy as np
import math

# =============================================================================
# 1) Helper Functions
# =============================================================================
def kl_std_normal(mean_squared, var):
    """
    Compute the KL divergence between a Gaussian q ~ N(0, var) and the 
    standard normal N(0, 1):
        KL(q || N(0,1)) = 0.5 * (var + mean_squared - log(var) - 1)
    """
    var_clamped = tf.maximum(var, 1e-15)
    return 0.5 * (var + mean_squared - tf.math.log(var_clamped) - 1.0)

def safe_log_softmax(logits, axis=-1):
    """
    Numerically stable log softmax.
    """
    return tf.nn.log_softmax(logits, axis=axis)

# =============================================================================
# 2) Schedule Classes
# =============================================================================
class FixedLinearSchedule(tf.keras.layers.Layer):
    """
    Fixed linear schedule: gamma(t) = GammaMin + (GammaMax - GammaMin) * t,
    where t is a float in [0,1].
    """
    def __init__(self, GammaMin, GammaMax, **kwargs):
        super().__init__(**kwargs)
        self.GammaMin = GammaMin
        self.GammaMax = GammaMax

    def call(self, t):
        return self.GammaMin + (self.GammaMax - self.GammaMin) * t

    def get_config(self):
        config = super().get_config()
        config.update({
            "GammaMin": self.GammaMin,
            "GammaMax": self.GammaMax
        })
        return config

class LearnedLinearSchedule(tf.keras.Model):
    """
    Learnable linear schedule: gamma(t) = b + |w| * t.
    """
    def __init__(self, GammaMin, GammaMax, **kwargs):
        super().__init__(**kwargs)
        # b = GammaMin, w = GammaMax - GammaMin
        self.b = tf.Variable(GammaMin, trainable=True, dtype=tf.float32, name="b")
        self.w = tf.Variable(GammaMax - GammaMin, trainable=True, dtype=tf.float32, name="w")

    def call(self, t):
        return self.b + tf.abs(self.w) * t

    def get_config(self):
        # 변수값을 numpy로 추출하여 config에 저장합니다.
        config = {"GammaMin": float(self.b.numpy()),
                  "GammaMax": float(self.b.numpy() + tf.abs(self.w).numpy())}
        return config

# =============================================================================
# 3) FiLM Layer
# =============================================================================
class FiLM(tf.keras.layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM) layer for condition-based modulation.
    
    Given an input tensor (B, T, C) and a condition vector (B, cond_dim),
    the layer computes per-channel scaling (gamma) and shift (beta) factors.
    """
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.gamma_layer = tf.keras.layers.Dense(channels, activation="linear")
        self.beta_layer  = tf.keras.layers.Dense(channels, activation="linear")

    def call(self, inputs, condition):
        # condition: shape (B, cond_dim)
        gamma = self.gamma_layer(condition)[:, None, :]  # (B, 1, C)
        beta  = self.beta_layer(condition)[:, None, :]     # (B, 1, C)
        return gamma * inputs + beta

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

# =============================================================================
# 4) Custom 1D Dilated Convolution Layer
# =============================================================================
class DilatedConv1d(tf.keras.layers.Layer):
    """
    1D dilated convolution.
    
    Args:
        in_channels: int, number of input channels.
        out_channels: int, number of output channels.
        kernel_size: int, size of the convolution kernel.
        dilation_rate: int, dilation rate.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.kernel_size, self.in_channels, self.out_channels),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(1, 1, self.out_channels),
            initializer='zeros',
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        conv = tf.nn.conv1d(inputs, self.kernel, stride=1, padding='SAME', dilations=self.dilation_rate)
        return conv + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate
        })
        return config

# =============================================================================
# 5) WaveNet Block with FiLM Integration
# =============================================================================
class Block(tf.keras.Model):
    """
    WaveNet block with FiLM modulation.
    
    Args:
        channels: int, number of channels.
        kernel_size: int, size of the convolution kernel.
        dilation: int, dilation rate.
        last: bool, if True, no residual connection is computed.
    """
    def __init__(self, channels, kernel_size, dilation, last=False, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.last = last

        # Project diffusion step embedding to channels.
        self.proj_embed = tf.keras.layers.Dense(channels)
        # FiLM modulation.
        self.film = FiLM(channels)
        # Dilated convolution doubling channels for gated activation.
        self.conv = DilatedConv1d(channels, channels * 2, kernel_size, dilation)
        if not last:
            self.proj_res = tf.keras.layers.Conv1D(channels, kernel_size=1)
        self.proj_skip = tf.keras.layers.Conv1D(channels, kernel_size=1)

    def call(self, inputs, embedding, condition):
        # Add diffusion embedding (broadcast along time axis).
        emb = self.proj_embed(embedding)  # shape: (B, C)
        x = inputs + emb[:, None, :]      # shape: (B, T, C)
        
        # Apply FiLM modulation.
        x = self.film(x, condition)
        # Apply dilated convolution.
        x = self.conv(x)  # shape: (B, T, 2*C)
        
        # Gated activation.
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate  # shape: (B, T, C)
        
        # Residual and skip connections.
        if not self.last:
            residual = (self.proj_res(x) + inputs) / math.sqrt(2.0)
        else:
            residual = None
            
        skip = self.proj_skip(x)
        return residual, skip

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "last": self.last,
        })
        return config

# =============================================================================
# 6) WaveNet Model
# =============================================================================
class WaveNet(tf.keras.Model):
    """
    WaveNet model with diffusion-step embeddings and FiLM modulation.
    
    Args:
        config: dict, configuration dictionary containing keys such as:
          "Channels", "Iter", "EmbeddingProj", "EmbeddingLayers",
          "NumLayers", "NumCycles", "DilationRate", "KernelSize",
          "EmbeddingSize", "EmbeddingFactor".
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Input projection: expected input shape (B, T, 1).
        self.proj = tf.keras.layers.Conv1D(config['Channels'], kernel_size=1)
        
        # Precompute sinusoidal embeddings for diffusion steps.
        self.embed = self._embedding(config['Iter'])
        
        # List of dense layers to refine the diffusion embedding.
        self.proj_embed = [tf.keras.layers.Dense(config['EmbeddingProj']) 
                           for _ in range(config['EmbeddingLayers'])]
        
        # Build WaveNet blocks.
        self.blocks = []
        layers_per_cycle = config['NumLayers'] // config['NumCycles']
        for i in range(config['NumLayers']):
            dilation = config['DilationRate'] ** (i % layers_per_cycle)
            is_last = (i == config['NumLayers'] - 1)
            self.blocks.append(Block(config['Channels'], config['KernelSize'], dilation, last=is_last))
        
        # Output projection layers.
        self.proj_out = [
            tf.keras.layers.Conv1D(config['Channels'], kernel_size=1, activation=tf.nn.relu),
            tf.keras.layers.Conv1D(1, kernel_size=1)
        ]

    def call(self, signal, timestep, condition):
        """
        Forward pass.
        
        Args:
            signal: Tensor of shape (B, T, 1) -- input signal.
            timestep: Tensor of shape (B,) -- diffusion timesteps.
            condition: Tensor of shape (B, cond_dim) -- condition input.
            
        Returns:
            Tensor of shape (B, T, 1) -- predicted noise.
        """
        x = tf.nn.relu(self.proj(signal)) 
        
        # Gather diffusion-step embedding.
        embed = tf.gather(self.embed, timestep - 1)  # shape: (B, EmbeddingSize)
        for proj in self.proj_embed:
            embed = tf.nn.swish(proj(embed))
        
        # Pass through WaveNet blocks and accumulate skip connections.
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x, embed, condition)
            skip_connections.append(skip)
        
        out = tf.add_n(skip_connections) / math.sqrt(len(self.blocks))
        for proj in self.proj_out:
            out = proj(out)
        return out # [B, T, 1]

    def _embedding(self, num_steps):
        """
        Generate sinusoidal embeddings for diffusion timesteps.
        
        Args:
            num_steps: int, number of diffusion steps.
        Returns:
            Tensor of shape (num_steps, EmbeddingSize).
        """
        half_dim = self.config['EmbeddingSize'] // 2
        linspace = tf.linspace(0.0, 1.0, half_dim)
        exp_term = tf.pow(10.0, linspace * self.config['EmbeddingFactor'])
        timesteps = tf.cast(tf.range(1, num_steps + 1), tf.float32)
        comp = timesteps[:, None] * exp_term[None, :]
        emb = tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)
        return emb

    def get_config(self):
        config = super().get_config()
        config.update({"config": self.config})
        return config

# =============================================================================
# 7) Variational Diffusion Model (VDM)
# =============================================================================
class VDM(tf.keras.Model):
    """
    Variational Diffusion Model for 1D signals with an additional 2D condition input.

    Args:
        cfg: dict, model configuration (e.g., number of diffusion steps, noise schedule, etc.).
        signal_shape: tuple, input signal shape (T, C).
    """
    def __init__(self, cfg, signal_shape=(1024, 1), **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.signal_shape = signal_shape=(cfg['SigDim'], 1)
        self.wavenet = WaveNet(cfg)
        
        # Noise schedule: Use either a fixed linear schedule or a learnable linear schedule.
        if cfg["NoiseSchedule"] == "fixed_linear":
            self.gamma = FixedLinearSchedule(cfg["GammaMin"], cfg["GammaMax"])
        elif cfg["NoiseSchedule"] == "learned_linear":
            self.gamma = LearnedLinearSchedule(cfg["GammaMin"], cfg["GammaMax"])
        else:
            raise ValueError(f"Unknown noise schedule {cfg['NoiseSchedule']}")

    def sample_q_t_0(self, x, t_float, noise=None, gamma_t=None):
        """
        Sample from q(x_t | x_0).

        Args:
            x: Tensor of shape (B, T, 1), original signal.
            t_float: Tensor of shape (B,), time values in the range [0,1].
            noise: Optional, noise tensor.
            gamma_t: Optional, precomputed gamma(t).
        
        Returns:
            x_t, gamma_t, used noise.
        """
        if gamma_t is None:
            gamma_t = self.gamma(t_float)  # [B]
        alpha_t = tf.sqrt(tf.nn.sigmoid(-gamma_t))
        sigma_t = tf.sqrt(tf.nn.sigmoid(gamma_t))
        if noise is None:
            noise = tf.random.normal(tf.shape(x))
        alpha_t_exp = tf.reshape(alpha_t, [-1, 1, 1])
        sigma_t_exp = tf.reshape(sigma_t, [-1, 1, 1])
        x_t = alpha_t_exp * x + sigma_t_exp * noise
        return x_t, gamma_t, noise

    def log_probs_x_z0(self, x=None, z_0=None, min_sigma=1e-7, RecLossType='logP'):
        """
        Compute log p(x | z_0) under a continuous Gaussian assumption.
        """
        gamma_0 = self.gamma(tf.constant([0.0], dtype=tf.float32))
        # alpha_0, sigma_0 from gamma_0
        alpha_0 = tf.sqrt(tf.nn.sigmoid(-gamma_0))
        sigma_0 = tf.sqrt(tf.nn.sigmoid(gamma_0))
        sigma_0 = tf.maximum(sigma_0, min_sigma)
    
        if x is not None and z_0 is None:
            # x -> z_0 (forward diffusion step)
            # z_0 = alpha_0*x + sigma_0*noise
            noise = tf.random.normal(tf.shape(x))
            x_pred = x + sigma_0 * noise / alpha_0
        elif z_0 is not None and x is not None:
            # z_0 -> x (denoising step)
            # x_pred = z_0 / alpha_0
            x_pred = z_0 / alpha_0
        else:
            raise ValueError("Provide (x, z_0=None) or (x, z_0) for reconstruction.")
    
        if RecLossType == 'logP':
            # Gaussian log-likelihood:
            # log p(x | x_pred) = -0.5 * ((x - x_pred)^2 / sigma_0^2) - log(sigma_0) - 0.5 * log(2*pi)
            # sum over time dimension
            loss = -0.5 * tf.square((x - x_pred) / sigma_0) - tf.math.log(sigma_0) - 0.5 * tf.math.log(2. * np.pi)
            # sum over (T,1) => shape [B]
            loss = -tf.reduce_sum(loss, axis=[1, 2])
        elif RecLossType == 'mse':
            # MSE version
            loss = tf.reduce_mean(tf.square(x - x_pred), axis=[1, 2])
        else:
            raise ValueError("RecLossType must be either 'logP' or 'mse'.")
        return loss

    def sample_p_s_t(self, z, t, s, condition, clip_samples):
        """
        Sample from p(z_s | z_t).
        Args:
            z: Tensor of shape [B, T, 1], current latent.
            t: Tensor of shape [B] (integer current timestep).
            s: Tensor of shape [B] (integer next timestep).
            condition: Tensor of shape [B, cond_dim].
            clip_samples: Boolean.
        Returns:
            z_next of shape [B, T, 1].
        """
        t_float = (tf.cast(t, tf.float32) - 1) / (tf.cast(self.cfg['Iter'] - 1, tf.float32))
        s_float = (tf.cast(s, tf.float32) - 1) / (tf.cast(self.cfg['Iter'] - 1, tf.float32))
        gamma_t = self.gamma(t_float)
        gamma_s = self.gamma(s_float)
        c = -tf.math.expm1(gamma_s - gamma_t)
        alpha_t = tf.sqrt(tf.nn.sigmoid(-gamma_t))
        alpha_s = tf.sqrt(tf.nn.sigmoid(-gamma_s))
        sigma_t = tf.sqrt(tf.nn.sigmoid(gamma_t))
        sigma_s = tf.sqrt(tf.nn.sigmoid(gamma_s))

        pred_noise = self.wavenet(z, t, condition)

        alpha_t_exp = tf.reshape(alpha_t, [-1, 1, 1])
        alpha_s_exp = tf.reshape(alpha_s, [-1, 1, 1])
        sigma_t_exp = tf.reshape(sigma_t, [-1, 1, 1])
        sigma_s_exp = tf.reshape(sigma_s, [-1, 1, 1])
        c_exp = tf.reshape(c, [-1, 1, 1])

        if clip_samples:
            x_start = (z - sigma_t_exp * pred_noise) / alpha_t_exp
            mean = alpha_s_exp * (z * (1.0 - c_exp) / alpha_t_exp + c_exp * x_start)
        else:
            mean = (alpha_s_exp / alpha_t_exp) * (z - c_exp * sigma_t_exp * pred_noise)
        
        scale = sigma_s_exp * tf.sqrt(c_exp)
        z_next = mean + scale * tf.random.normal(tf.shape(z))
        return z_next

    def compute_loss(self, x, condition, noise=None):
        """
        Compute the total loss (diffusion loss, latent loss, reconstruction loss).
        """
        # Sample discrete timesteps uniformly from {1,..., Iter}
        batch_size = tf.shape(x)[0]
        timestep = tf.random.uniform([batch_size], minval=1, maxval=self.cfg['Iter'] + 1, dtype=tf.int32)
        # Convert to float in [0,1]
        t_float = tf.cast(timestep - 1, tf.float32) / tf.cast(self.cfg['Iter'] - 1, tf.float32)

        # 1) Diffusion Loss
        with tf.GradientTape() as tape_inner:
            tape_inner.watch(t_float)
            gamma_t = self.gamma(t_float)
        gamma_grad = tape_inner.gradient(gamma_t, t_float)

        # q(x_t | x_0) : Diffusion process
        x_t, gamma_t, noise_used = self.sample_q_t_0(x, t_float, noise, gamma_t=gamma_t)

        # Predict noise
        pred_noise = self.wavenet(x_t, timestep, condition)
        squared_diff = tf.reduce_sum(tf.square(pred_noise - noise_used), axis=[1, 2])

        # Weighted by gamma'(t)
        diffusion_loss = 0.5 * squared_diff * gamma_grad  # shape [B]

        # 2) Latent loss (KL divergence) from x1 to standard normal
        gamma_1 = self.gamma(tf.constant([1.0], dtype=tf.float32))
        sigma_1_sq = tf.nn.sigmoid(gamma_1)
        alpha_1_sq = 1.0 - sigma_1_sq
        mean_sq = alpha_1_sq * tf.square(x)
        kl_val = kl_std_normal(mean_sq, sigma_1_sq)
        latent_loss = tf.reduce_sum(kl_val, axis=[1, 2])   # shape [B]

        # 3) Reconstruction loss: continuous (Gaussian NLL)
        recons_loss = self.log_probs_x_z0(x, z_0=None, min_sigma=self.cfg['SigmaMin'])  # shape [B]
        total_loss_per_sample = diffusion_loss + latent_loss + recons_loss
        total_loss = tf.reduce_mean(total_loss_per_sample)
        return total_loss

    def call(self, inputs, training=False, noise=None):
        """
        Returns loss during training mode, and sampled results during inference mode.
    
        Args:
            inputs: tuple (x, condition) where
                x: Tensor of shape (B, T), input signal.
                condition: Tensor of shape (B, cond_dim), conditioning input.
            training: bool, whether the model is in training mode.
    
        Returns:
            Loss value if training=True, otherwise a sampled signal.
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
            inputs = inputs[0]
        x, condition = inputs[0][...,None], inputs[1]
        
        if training:
            return self.compute_loss(x, condition, noise)
        else:
            batch_size = tf.shape(x)[0]
            return self.sample(batch_size, condition, n_sample_steps=self.cfg['Iter'], clip_samples=False)

    @tf.function
    def sample(self, batch_size, condition, n_sample_steps=50, clip_samples=False):
        """
        Performs ancestral sampling for inference.
    
        Args:
            batch_size: int, number of samples to generate.
            condition: Tensor of shape (B, cond_dim), conditioning input.
            n_sample_steps: int, number of sampling steps.
            clip_samples: bool, whether to clip the sampled values within the range [-1,1].
    
        Returns:
            Tensor of shape (B, T, C), the generated signal.
        """
        T, C = self.signal_shape
        # Start with Gaussian noise in z space
        z = tf.random.normal(shape=(batch_size, T, C))

        # Sequence of integer timesteps from Iter down to 1
        # e.g., if Iter=1000, then t goes 1000 -> 999 -> ... -> 1, but limited by n_sample_steps
        t_vals = tf.linspace(tf.cast(self.cfg['Iter'], tf.float32), 1.0, n_sample_steps + 1)
        
        for i in tf.range(n_sample_steps):
            t_ = t_vals[i]
            s_ = t_vals[i + 1]
            t_batch = tf.cast(tf.ones((batch_size,)) * t_, tf.int32)
            s_batch = tf.cast(tf.ones((batch_size,)) * s_, tf.int32)
            z = self.sample_p_s_t(z, t_batch, s_batch, condition, clip_samples)

        # Convert final z -> x via alpha_0 if you want the pure "z_0 -> x" transform:
        alpha_0 = tf.sqrt(tf.nn.sigmoid(-self.gamma(tf.constant([0.0], dtype=tf.float32))))
        x_out = z / alpha_0

        # Optionally clip to [-1,1]    
        if clip_samples:
            x_out = tf.clip_by_value(x_out, -1.0, 1.0)
    
        return x_out  # shape [B, T, C]

    def train_step(self, data):
        """
        Custom training step for `keras.wavenet.fit()`.
        """
        if isinstance(data, (list, tuple)) and len(data) == 1:
            data = data[0]
        x, condition = data[0][...,None], data[1]
        
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, condition)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        """
        Custom evaluation/validation step.
        """
        if isinstance(data, (list, tuple)) and len(data) == 1:
            data = data[0]
        x, condition = data[0][...,None], data[1]
        loss = self.compute_loss(x, condition)
        return {"loss": loss}

    def get_config(self):
        config = super().get_config()
        config.update({
            "cfg": self.cfg,
            "signal_shape": self.signal_shape,
        })
        return config
