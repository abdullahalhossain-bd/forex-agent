# ml/timesformer_block.py — TimeSformer (spatiotemporal transformer) block
# =============================================================================
# Ported from: https://github.com/zeroxt32/Forex-Expert-Advisor-Python/blob/master/T32_v5.py
# Original: TimeSformerBlock class (lines 795-825)
# Original author: zeroxt32 — MIT license
#
# A transformer block designed for SPATIOTEMPORAL data (video-like sequences
# of images, e.g., consecutive chart screenshots). Implements:
#
#   1. Multi-head self-attention across the temporal dimension
#   2. Feedforward network (2-layer MLP with ReLU)
#   3. Layer normalization + residual connections (pre-norm architecture)
#
# This is a clean, standalone implementation of the TimeSformer block from
# the original paper ("TimeSformer: Is Space-Time Attention All You Need for
# Video Understanding?"). It can be used as a building block in any Keras
# model that processes sequences of spatial observations.
#
# Architecture (per block):
#   x → LayerNorm → MultiHeadAttention → + x (residual)
#     → LayerNorm → Dense(hidden*4, relu) → Dense(hidden) → + x (residual)
#
# Requires TensorFlow/Keras. The module gracefully degrades to a no-op
# stub if TF is not installed (for environments that only need the RL
# reward functions without the neural network).
# =============================================================================

from __future__ import annotations

from utils.logger import get_logger

log = get_logger("timesformer_block")

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    log.info("TensorFlow not installed — TimeSformerBlock will not be available. "
             "Install with: pip install tensorflow")


if _HAS_TF:

    class TimeSformerBlock(tf.keras.layers.Layer):
        """
        Spatiotemporal transformer block with multi-head self-attention.

        Parameters
        ----------
        hidden_dim : int
            Dimension of the hidden representations (default 256).
        num_heads : int
            Number of attention heads (default 8).

        Usage
        -----
            block = TimeSformerBlock(hidden_dim=256, num_heads=8)
            output = block(input_tensor)  # input: (batch, seq_len, hidden_dim)

        Or as part of a Sequential model:
            model = Sequential([
                TimeDistributed(cnn, input_shape=(seq_len, *img_shape)),
                TimeSformerBlock(hidden_dim=256, num_heads=8),
                Dense(128, activation='relu'),
                Dense(action_size, activation='softmax'),
            ])
        """

        def __init__(self, hidden_dim: int = 256, num_heads: int = 8, **kwargs):
            super().__init__(**kwargs)
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads

            # Self-attention layer
            self.self_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=hidden_dim
            )

            # Feedforward network (2-layer MLP)
            self.feedforward = tf.keras.Sequential([
                tf.keras.layers.Dense(hidden_dim * 4, activation='relu'),
                tf.keras.layers.Dense(hidden_dim)
            ])

            # Layer normalization (pre-norm architecture)
            self.norm1 = tf.keras.layers.LayerNormalization()
            self.norm2 = tf.keras.layers.LayerNormalization()

        def call(self, x):
            """
            Forward pass.

            Parameters
            ----------
            x : tensor of shape (batch, seq_len, hidden_dim)

            Returns
            -------
            tensor of same shape after self-attention + feedforward + residuals.
            """
            # Pre-norm self-attention with residual
            norm_x = self.norm1(x)
            attention_output = self.self_attention(norm_x, norm_x)
            x = x + attention_output

            # Pre-norm feedforward with residual
            norm_x = self.norm2(x)
            feedforward_output = self.feedforward(norm_x)
            x = x + feedforward_output

            return x

        def get_config(self):
            config = super().get_config()
            config.update({
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
            })
            return config

else:
    # Stub when TF is not available
    class TimeSformerBlock:  # type: ignore[no-redef]
        """Stub — install TensorFlow to use the real TimeSformerBlock."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow"
            )


# ── Factory: build a full actor/critic model with TimeSformer ────────────────

def build_visual_actor(
    state_size: tuple,
    action_size: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    learning_rate: float = 1e-5,
) -> "tf.keras.Model":
    """
    Build an actor model: 3D CNN → TimeSformer → Dense → softmax actions.

    Parameters
    ----------
    state_size : (seq_len, height, width, channels) — e.g., (4, 224, 224, 3)
    action_size : number of discrete actions (e.g., 3 for BUY/SELL/HOLD)
    hidden_dim : transformer hidden dimension
    num_heads : attention heads
    learning_rate : Adam learning rate

    Returns
    -------
    Compiled Keras model.
    """
    if not _HAS_TF:
        raise ImportError("TensorFlow required")

    # 3D CNN feature extractor (processes each frame in the sequence)
    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), strides=(1, 4, 4), padding="same",
                                input_shape=state_size),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'),
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same"),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'),
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same"),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(cnn, input_shape=state_size),
        TimeSformerBlock(hidden_dim=hidden_dim, num_heads=num_heads),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='softmax'),
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


def build_visual_critic(
    state_size: tuple,
    action_size: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    learning_rate: float = 1e-5,
) -> "tf.keras.Model":
    """
    Build a critic model: same architecture as actor but outputs a single
    Q-value (linear activation).
    """
    if not _HAS_TF:
        raise ImportError("TensorFlow required")

    cnn = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, (3, 3, 3), strides=(1, 4, 4), padding="same",
                                input_shape=state_size),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'),
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same"),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'),
        tf.keras.layers.Conv3D(64, (3, 3, 3), padding="same"),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(cnn, input_shape=state_size),
        TimeSformerBlock(hidden_dim=hidden_dim, num_heads=num_heads),
        tf.keras.layers.Dense(hidden_dim, activation='relu'),
        tf.keras.layers.Dense(hidden_dim // 2, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='softmax'),  # shared with actor
        tf.keras.layers.Dense(1, activation='linear'),  # value output
    ])
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


# ── Smoke test (only if TF is available) ─────────────────────────────────────
if __name__ == "__main__":
    if not _HAS_TF:
        print("TensorFlow not installed — smoke test skipped.")
        print("Install with: pip install tensorflow")
    else:
        # Test the TimeSformerBlock in isolation
        import numpy as np

        block = TimeSformerBlock(hidden_dim=64, num_heads=4)
        # (batch, seq_len, hidden_dim)
        x = tf.random.normal((2, 4, 64))
        y = block(x)
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {y.shape}")
        assert x.shape == y.shape, "TimeSformerBlock should preserve shape"

        # Test model building (small dimensions for speed)
        print("\nBuilding actor model (small)...")
        actor = build_visual_actor(
            state_size=(4, 32, 32, 3),  # 4 frames of 32×32 RGB
            action_size=3,               # BUY/SELL/HOLD
            hidden_dim=32,
            num_heads=4,
        )
        actor.summary(print_fn=print)

        # Forward pass
        dummy_state = tf.random.normal((1, 4, 32, 32, 3))
        pred = actor(dummy_state)
        print(f"\nActor prediction shape: {pred.shape}")
        assert pred.shape[-1] == 3  # 3 actions

        # Critic
        print("\nBuilding critic model (small)...")
        critic = build_visual_critic(
            state_size=(4, 32, 32, 3), action_size=3,
            hidden_dim=32, num_heads=4,
        )
        value = critic(dummy_state)
        print(f"Critic value shape: {value.shape}")

        print("\nTimeSformerBlock + visual actor/critic smoke test passed.")
