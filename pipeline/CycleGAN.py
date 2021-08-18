########################################################################################################################
# CycleGAN pipeline
########################################################################################################################
import logging
import os
import random
from time import time

import cv2
import numpy as np
import tensorflow as tf

from data_utils.DataSet2DUnpaired import DataSet2DUnpaired
from losses.gan import generator_loss_lsgan, cycle_consistency_loss, identity_loss, discriminator_loss_lsgan
from models.ResnetGenerator import ResnetGenerator
from models.ConvDiscriminator import ConvDiscriminator
from pipeline.Checkpoint import Checkpoint
from pipeline.LinearDecay import LinearDecay

__author__ = "c.magg"


class CycleGAN:

    def __init__(self, data_dir, tensorboard_dir, checkpoints_dir, save_model_dir, sample_dir,
                 seed=13375, d_step=1, sample_step=500,
                 cycle_loss_weight=10.0, identity_loss_weight=1.0,
                 dsize=(256, 256)):
        """
        CycleGan Pipeline
        :param data_dir: directory for data (needs to contain a training/test/validation folder
        :param tensorboard_dir: directory for tensorboard logging
        :param checkpoints_dir: directory for checkpoints
        :param save_model_dir: directory for saved models
        :param sample_dir: directory for samples
        :param seed: random seed
        :param d_step: discriminator will be trained each d_step
        :param sample_step: sample from validator will be created each sample_step
        :param cycle_loss_weight: weight for the cycle loss
        :param identity_loss_weight: weight for the identity loss
        :param dsize: image size
        """

        # random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # directories
        self.dir_data = data_dir
        self.dir_tb = tensorboard_dir
        self.dir_cktp = checkpoints_dir
        self.dir_save_model = save_model_dir if save_model_dir is not None else ""
        self.dir_sample = sample_dir
        if not os.path.isdir(sample_dir):
            logging.info(f"CycleGan: create {sample_dir}.")
            os.makedirs(sample_dir)

        # parameters
        self.cycle_loss_weight = cycle_loss_weight
        self.identity_loss_weight = identity_loss_weight
        self.d_step = d_step
        self.sample_step = sample_step
        self.dsize = dsize

        # data
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self._load_data()

        # generator
        self.G_S2T = ResnetGenerator(n_blocks=9, input_shape=(*self.dsize, 1)).generate_model()
        self.G_T2S = ResnetGenerator(n_blocks=9, input_shape=(*self.dsize, 1)).generate_model()

        # discriminator
        self.D_S = ConvDiscriminator(input_shape=(*self.dsize, 1)).generate_model()
        self.D_T = ConvDiscriminator(input_shape=(*self.dsize, 1)).generate_model()

        # optimizer
        self.G_optimizer = None
        self.D_optimizer = None
        self._set_optimizer()

        # fake image history
        self._pool_size = 50
        self._fake_S = np.zeros((self._pool_size, 1, 256, 256, 1))
        self._fake_T = np.zeros((self._pool_size, 1, 256, 256, 1))
        self._num_fake = 0

        # checkpoints and template
        self.template = "{4}/{5} in {6:.4f} sec - S_d_loss: {0:.5f} - T_d_loss: {1:.5f} - S2T_g_loss: {2:.5f} - T2S_g_loss: {3:.5f}"
        self.checkpoint = None
        self.train_summary_writer = None

    def _set_optimizer(self, total_steps=50000, step_decay=25000):
        """
        Set optimizer with linear LR scheduler
        """
        G_lr_scheduler = LinearDecay(0.0002, total_steps, step_decay)
        D_lr_scheduler = LinearDecay(0.0002, total_steps, step_decay)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=0.5)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=0.5)

    def _load_data(self):
        """
        Load data from training/validation/test folder with batch size 1, no augm and pixel value range [-1,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, unpaired
        Validation data - without shuffle, paired
        Test data - without shuffle, paired
        """
        logging.info("CycleGAN: loading data ...")
        self.train_set = DataSet2DUnpaired(os.path.join(self.dir_data, "training"), input_data=["t1"],
                                           input_name=["image"], output_data="t2", output_name="generated_t2",
                                           batch_size=1, shuffle=True, p_augm=0.0, alpha=-1, beta=1, use_filter="vs",
                                           dsize=self.dsize)
        self.val_set = DataSet2DUnpaired(os.path.join(self.dir_data, "validation"), input_data=["t1"],
                                         input_name=["image"], output_data="t2", output_name="generated_t2",
                                         batch_size=1, shuffle=False, p_augm=0.0, alpha=-1, beta=1, use_filter="vs",
                                         dsize=self.dsize)
        self.val_set._unpaired = False
        self.val_set.reset()
        logging.info("CycleGAN: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))

        self.test_set = DataSet2DUnpaired(os.path.join(self.dir_data, "test"), input_data=["t1"],
                                          input_name=["image"], output_data="t2", output_name="generated_t2",
                                          batch_size=1, shuffle=False, p_augm=0.0, alpha=-1, beta=1, use_filter="vs",
                                          dsize=self.dsize)
        self.test_set._unpaired = False
        self.test_set.reset()

        logging.info("CycleGAN: test {0}".format(len(self.test_set)))

    @tf.function
    def _train_generator(self, S, T):
        """
        Generator training consists of the following steps (always for both domains):
        1) generator training: G_S2T(S) = T_
        2) cycle: G_T2S(T_) = S_
        3) identity: G_T2S(S) = S_
        4) loss calculation:
            - adversarial loss: based on D decision
            - cycle consistency loss: how similar are S and S_ from generator cycle
            - identity loss: how similar are S and S_ from identity
        5) update gradients
        """
        with tf.GradientTape() as tape:
            # generator
            S2T = self.G_S2T(S, training=True)
            T2S = self.G_T2S(T, training=True)

            # generator cycle
            S2T2S = self.G_T2S(S2T, training=True)
            T2S2T = self.G_S2T(T2S, training=True)

            # generator identity
            S2S = self.G_T2S(S, training=True)
            T2T = self.G_S2T(T, training=True)

            # discriminator
            S2T_d_logits = self.D_T(S2T, training=True)
            T2S_d_logits = self.D_S(T2S, training=True)

            # generator loss
            S2T_g_loss = generator_loss_lsgan(S2T_d_logits)
            T2S_g_loss = generator_loss_lsgan(T2S_d_logits)

            # cycle consistency loss
            S2T2S_c_loss = cycle_consistency_loss(S, S2T2S)
            T2S2T_c_loss = cycle_consistency_loss(T, T2S2T)

            # identity loss
            S2S_id_loss = identity_loss(S, S2S)
            T2T_id_loss = identity_loss(T, T2T)

            # total loss
            G_loss = (S2T_g_loss + T2S_g_loss) + (S2T2S_c_loss + T2S2T_c_loss) * self.cycle_loss_weight + (
                    S2S_id_loss + T2T_id_loss) * self.identity_loss_weight

        # calc and update gradients
        G_grad = tape.gradient(G_loss, self.G_S2T.trainable_variables + self.G_T2S.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G_S2T.trainable_variables + self.G_T2S.trainable_variables))

        return S2T, T2S, {'S2T_g_loss': S2T_g_loss,
                          'T2S_g_loss': T2S_g_loss,
                          'S2T2S_cycle_loss': S2T2S_c_loss,
                          'T2S2T_cycle_loss': T2S2T_c_loss,
                          'S2S_id_loss': S2S_id_loss,
                          'T2T_id_loss': T2T_id_loss}

    @tf.function
    def _train_discriminator(self, S, T, S2T, T2S):
        """
        Discriminator training consists of the following steps (always for both domains):
        1) discriminator training real: D_S(S) = S_real
        2) discriminator training fake: D_S(T2S) = S_fake
        3) loss calculation:
            - adversarial loss: compare the real to ones and fakes to zeros and weight the loss
        4) update gradients
        """
        with tf.GradientTape() as tape:
            S_d_logits = self.D_S(S, training=True)  # real
            T2S_d_logits = self.D_S(T2S, training=True)  # fake
            T_d_logits = self.D_T(T, training=True)  # real
            S2T_d_logits = self.D_T(S2T, training=True)  # fake

            S_d_loss = discriminator_loss_lsgan(S_d_logits, T2S_d_logits)
            T_d_loss = discriminator_loss_lsgan(T_d_logits, S2T_d_logits)

            D_loss = S_d_loss + T_d_loss

        # calc and update gradients
        D_grad = tape.gradient(D_loss, self.D_S.trainable_variables + self.D_T.trainable_variables)
        self.D_optimizer.apply_gradients(
            zip(D_grad, self.D_S.trainable_variables + self.D_T.trainable_variables))

        return {'S_d_loss': S_d_loss,
                'T_d_loss': T_d_loss}

    def train_step(self, S, T, step, D_loss_dict):
        """
        One training step:
        1) train generators
        2) chose fake images from a pool (and fill pool with new generated images)
        3) train discriminator (for special number of steps, eg every 10th step, discriminator is trained)
        """
        # train generator
        S2T, T2S, G_loss_dict = self._train_generator(S, T)

        # chose A2B, B2A from history
        S2T_fake = self._fake_image_history(self._num_fake, S2T, self._fake_T)
        T2S_fake = self._fake_image_history(self._num_fake, T2S, self._fake_S)
        self._num_fake += 1

        # train descriminator
        if step % self.d_step == 0:
            D_loss_dict = self._train_discriminator(S, T, S2T_fake, T2S_fake)

        return G_loss_dict, D_loss_dict

    def _fake_image_history(self, num_fakes, fake, fake_pool):
        """
        Create a pool of fake images for given history length.
        taken from https://github.com/cchen-cc/SIFA/blob/be5b792ecb7ed85f533bbb91223a7278e969b12d/main.py#L294
        """
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def _init_summary_file_writer(self, directory=os.path.join("logs", "train")):
        """
        Initialize the tensorboard summary file writer.
        """
        logging.info("CycleGAN: set up summary file writer with directory {}.".format(directory))
        self.train_summary_writer = tf.summary.create_file_writer(directory)

    def _init_checkpoint(self, directory=os.path.join("tmp", 'checkpoints'), restore=True):
        """
        Initialize the model checkpoints.
        """
        logging.info("CycleGAN: set up checkpoints with directory {}.".format(directory))
        self.checkpoint = Checkpoint(dict(G_S2T=self.G_S2T,
                                          G_T2S=self.G_T2S,
                                          D_S=self.D_S,
                                          D_T=self.D_T,
                                          G_optimizer=self.G_optimizer,
                                          D_optimizer=self.D_optimizer),
                                     directory,
                                     max_to_keep=3)
        if restore:
            try:  # restore checkpoint including the epoch counter
                self.checkpoint.restore().assert_existing_objects_matched()
            except Exception as e:
                print("CycleGAN: " + e)

    @staticmethod
    def _collect_losses(G_loss_dict, D_loss_dict, G_loss_dict_list, D_loss_dict_list):
        """
        Collect losses from dict with one value to dict with list of values per epoch.
        """
        for k, v in G_loss_dict.items():
            if type(G_loss_dict_list[k]) == list:
                G_loss_dict_list[k].append(v.numpy())
            else:
                G_loss_dict_list[k] = [v.numpy()]
        for k, v in D_loss_dict.items():
            if type(D_loss_dict_list[k]) == list:
                D_loss_dict_list[k].append(v.numpy())
            else:
                D_loss_dict_list[k] = [v.numpy()]
        return G_loss_dict_list, D_loss_dict_list

    def _save_models(self):
        """
        Save models to SavedModel
        """
        logging.info("CycleGAN: save models to {}".format(self.dir_save_model))
        self.G_S2T.save(os.path.join(self.dir_save_model, "G_S2T"))
        self.G_T2S.save(os.path.join(self.dir_save_model, "G_T2S"))
        self.D_S.save(os.path.join(self.dir_save_model, "D_S"))
        self.D_T.save(os.path.join(self.dir_save_model, "D_T"))

    @tf.function
    def sample(self, S, T):
        """
        Generate samples from current generators.
        """
        S2T = self.G_S2T(S, training=False)
        T2S = self.G_T2S(T, training=False)
        S2T2S = self.G_T2S(S2T, training=False)
        T2S2T = self.G_S2T(T2S, training=False)
        return S2T, T2S, S2T2S, T2S2T

    def train(self, epochs=50, data_nr=None, restore=True, step_decay=None):
        """
        Train CycleGAN pipeline:
        1) initialize local variables
        2) initialize tensorboard file writer and checkpoints
        3) initialize optimizer with LR scheduler
        4) for each epoch:
            for each step:
                draw next batch from training data
                perform training step
                collect the losses
                if sample generation step: generate sample
            write tensorboard summary
            save checkpoint
        5) save model
        """
        logging.info("CycleGAN: set up training.")

        G_loss_dict_list = {k: 0 for k in ['S2T_g_loss', 'T2S_g_loss', 'S2T2S_cycle_loss', 'T2S2T_cycle_loss',
                                           'S2S_id_loss', 'T2T_id_loss']}
        D_loss_dict_list = {k: 0 for k in ['S_d_loss', 'T_d_loss']}
        if data_nr is None or data_nr > len(self.train_set):
            data_nr = len(self.train_set)
        self._init_summary_file_writer(self.dir_tb)
        self._init_checkpoint(self.dir_cktp, restore=restore)
        if step_decay is None:
            step_decay = epochs // 2
        self._set_optimizer((epochs+1)*data_nr, (step_decay+1)*data_nr)
        sample_counter = 0

        D_loss_dict = {'S_d_loss': tf.constant(100.0),
                       'T_d_loss': tf.constant(100.0)}

        logging.info("CycleGAN: start training.")
        for epoch in range(epochs + 1):
            total_time_per_epoch = 0
            print("Epoch {0}/{1}".format(epoch, epochs))
            for idx in range(data_nr):
                S, T = self.train_set[idx]
                S = S["image"]
                T = T["generated_t2"]
                start = time()
                G_loss_dict, D_loss_dict = self.train_step(S, T, idx, D_loss_dict)
                elapsed = time() - start
                total_time_per_epoch += elapsed
                print(self.template.format(D_loss_dict["S_d_loss"],
                                           D_loss_dict["T_d_loss"],
                                           G_loss_dict["S2T_g_loss"],
                                           G_loss_dict["T2S_g_loss"],
                                           idx, data_nr - 1,
                                           total_time_per_epoch),
                      end="\r")
                self.train_set.reset()
                G_loss_dict_list, D_loss_dict_list = self._collect_losses(G_loss_dict, D_loss_dict, G_loss_dict_list,
                                                                          D_loss_dict_list)
                if idx % self.sample_step == 0:
                    A, B = self.val_set[sample_counter]
                    A = A["image"]
                    B = B["generated_t2"]
                    A2B, B2A, A2B2A, B2A2B = self.sample(A, B)
                    img = np.hstack(
                        np.concatenate([tf.expand_dims(A, -1), A2B, A2B2A, tf.expand_dims(B, -1), B2A, B2A2B], axis=0))
                    img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    cv2.imwrite(os.path.join(self.dir_sample, 'iter-%03d-%05d.jpg' % (epoch, idx)),
                                img)
                    sample_counter = sample_counter + 1
                    if sample_counter >= len(self.val_set):
                        sample_counter = 0

            with self.train_summary_writer.as_default():
                for k, v in G_loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                for k, v in D_loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                tf.summary.scalar("learning_rate",
                                  self.G_optimizer.learning_rate.current_learning_rate,
                                  step=epoch)
            print(self.template.format(np.mean(D_loss_dict_list["S_d_loss"]),
                                       np.mean(D_loss_dict_list["T_d_loss"]),
                                       np.mean(G_loss_dict_list["S2T_g_loss"]),
                                       np.mean(G_loss_dict_list["T2S_g_loss"]),
                                       data_nr - 1, data_nr - 1, total_time_per_epoch))
            self.checkpoint.save(epoch)

        self._save_models()
