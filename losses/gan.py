########################################################################################################################
# Container for GAN losses

# based on github repo https://github.com/bryanlimy/tf2-cyclegan and https://github.com/LynnHo/CycleGAN-Tensorflow-2
########################################################################################################################

import tensorflow as tf

__author__ = "c.magg"


def cycle_consistency_loss(real_samples, cycle_samples, lambda_cycle=1):
    return lambda_cycle * tf.reduce_mean(tf.keras.losses.mean_absolute_error(real_samples, cycle_samples))


def identity_loss(real_samples, identity_samples, lambda_identity=1):
    return lambda_identity * tf.reduce_mean(tf.keras.losses.mean_absolute_error(real_samples, identity_samples))


def generator_loss_lsgan(fake_samples):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.ones_like(fake_samples), fake_samples))


def discriminator_loss_lsgan(real_samples, fake_samples):
    real_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.ones_like(real_samples), real_samples))
    fake_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.zeros_like(fake_samples), fake_samples))
    return 0.5 * (real_loss + fake_loss)
