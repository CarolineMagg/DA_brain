########################################################################################################################
# SIFA pipeline
########################################################################################################################
import logging
import os
import random
from time import time

import cv2
import numpy as np
import tensorflow as tf

from data_utils.DataSet2DMixed import DataSet2DMixed
from losses.dice import DiceCoefficient, DiceLoss
from losses.gan import generator_loss_lsgan, cycle_consistency_loss, identity_loss, discriminator_loss_lsgan
from models.ResnetGenerator import ResnetGenerator
from models.ConvDiscriminator import ConvDiscriminator
from models.SIFADecoder import SIFADecoder
from models.SIFAEncoder import SIFAEncoder
from models.SIFASegmentation import SIFASegmentation
from pipeline.Checkpoint import Checkpoint

__author__ = "c.magg"


class SIFA:

    def __init__(self, data_dir, tensorboard_dir, checkpoints_dir, save_model_dir, sample_dir,
                 seed=13375, d_step=1, sample_step=500,
                 cycle_loss_weight=10.0, identity_loss_weight=1.0,
                 dsize=(256, 256)):
        """
        SIFA Pipeline
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
            logging.info(f"SIFA: create {sample_dir}.")
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
        self.G_S2T = ResnetGenerator(n_blocks=4, input_shape=(*self.dsize, 1), skip=True).generate_model()

        # segmentation
        self.segmentation1 = SIFASegmentation(input_shape=(32, 32, 256)).generate_model()
        self.segmentation2 = SIFASegmentation(input_shape=(32, 32, 256)).generate_model()

        # encoder & decoder
        self.encoder = SIFAEncoder(input_shape=(*self.dsize, 1)).generate_model_small()
        self.decoder = SIFADecoder(input_shape=(32, 32, 256), skip=True).generate_model()

        # discriminator
        self.D_T = ConvDiscriminator(input_shape=(*self.dsize, 1), dim=32).generate_model()
        self.D_S = ConvDiscriminator(input_shape=(*self.dsize, 1), dim=32).generate_model_aux()
        self.D_P = ConvDiscriminator(input_shape=(*self.dsize, 1), dim=32).generate_model()
        self.D_P2 = ConvDiscriminator(input_shape=(*self.dsize, 1), dim=32).generate_model()

        # optimizer
        self.D_S_optimizer = None
        self.D_T_optimizer = None
        self.D_P_optimizer = None
        self.D_P2_optimizer = None
        self.G_S2T_optimizer = None
        self.decoder_optimizer = None
        self.segmentation_optimizer = None
        self._set_optimizer()

        # fake image history
        self._pool_size = 50
        self._fake_S = np.zeros((self._pool_size, 1, 256, 256, 1))
        self._fake_T = np.zeros((self._pool_size, 1, 256, 256, 1))
        self._num_fake = 0

        # checkpoints and template
        self.template = "{3}/{4} in {5:.4f} sec - T_seg_loss: {0:.5f} - T_dice_loss: {1:.5f} - T_dice_coeff: {2:.5f}"
        self.checkpoint = None
        self.train_summary_writer = None

    def _set_optimizer(self):
        """
        Set optimizer with LR 0.0002
        """
        learning_rate = 0.0002
        self.D_S_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.D_T_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.D_P_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.D_P2_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.G_S2T_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
        self.segmentation_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    def _load_data(self):
        """
        Load data from training/validation/test folder with batch size 1, no augm and pixel value range [-1,1].
        Use only data where segmentation is available to ensure tumor presents.
        Training data - with shuffle, unpaired
        Validation data - without shuffle, paired
        Test data - without shuffle, paired
        """
        logging.info("SIFA: loading data ...")
        self.train_set = DataSet2DMixed(os.path.join(self.dir_data, "training"), input_data=["t1"],
                                        input_name=["image"], output_data=["t2", "vs"],
                                        output_name=["t2", "vs"],
                                        batch_size=1, shuffle=True, p_augm=0.0, alpha=-1, beta=1, use_filter="vs",
                                        dsize=self.dsize)
        self.val_set = DataSet2DMixed(os.path.join(self.dir_data, "validation"), input_data=["t1"],
                                      input_name=["image"], output_data=["t2", "vs"],
                                      output_name=["t2", "vs"],
                                      batch_size=1, shuffle=False, p_augm=0.0, alpha=-1, beta=1, use_filter="vs",
                                      dsize=self.dsize)
        self.val_set._unpaired = False
        self.val_set.reset()
        logging.info("SIFA: training {0}, validation {1}".format(len(self.train_set), len(self.val_set)))

        self.test_set = DataSet2DMixed(os.path.join(self.dir_data, "test"), input_data=["t1"],
                                       input_name=["image"], output_data=["t2", "vs"],
                                       output_name=["t2", "vs"],
                                       batch_size=1, shuffle=False, p_augm=0.0, alpha=-1, beta=1, use_filter="vs",
                                       dsize=self.dsize)
        self.test_set._unpaired = False
        self.test_set.reset()

        logging.info("SIFA: test {0}".format(len(self.test_set)))

    @tf.function
    def _train_generator_decoder(self, S, T):
        with tf.GradientTape(persistent=True) as tape:
            # S -> T -> latent -> S
            S2T = self.G_S2T([S, S], training=True)
            S2T_d_logits = self.D_T(S2T, training=True)
            # D_T result of fake T
            S2T_latent1, S2T_latent2 = self.encoder(S2T, training=True)
            # S2T2S - cycle image S
            S2T2S = self.decoder({"encoder": S2T_latent1, "tmp": S2T}, training=True)

            # T -> latent -> S
            T_latent1, T_latent2 = self.encoder(T, training=True)
            T2S = self.decoder({"encoder": T_latent1, "tmp": T}, training=True)
            # D_S result of fake S
            T2S_d_logits, T2S_d_logits_aux = self.D_S(T2S, training=True)
            # T2S2T - cycle image T
            T2S2T = self.G_S2T([T2S, T2S], training=True)

            # cycle consistency losses
            S2T2S_c_loss = cycle_consistency_loss(S, S2T2S)
            T2S2T_c_loss = cycle_consistency_loss(T, T2S2T)
            # gan losses
            S2T_g_loss = generator_loss_lsgan(S2T_d_logits)
            T2S_g_loss = generator_loss_lsgan(T2S_d_logits)

            # generator loss
            S_g_loss = S2T2S_c_loss + T2S2T_c_loss + T2S_g_loss
            T_g_loss = S2T2S_c_loss + T2S2T_c_loss + S2T_g_loss

        G_S2T_grad = tape.gradient(S_g_loss, self.G_S2T.trainable_variables)
        Decoder_grad = tape.gradient(T_g_loss, self.decoder.trainable_variables)
        self.G_S2T_optimizer.apply_gradients(
            zip(G_S2T_grad, self.G_S2T.trainable_variables))
        self.decoder_optimizer.apply_gradients(
            zip(Decoder_grad, self.decoder.trainable_variables))
        del tape

        loss_dict = {'S2T2S_c_loss': S2T2S_c_loss,
                     'T2S2T_c_loss': T2S2T_c_loss,
                     'S2T_g_loss': S2T_g_loss,
                     'T2S_g_loss': T2S_g_loss,
                     'S_g_loss': S_g_loss,
                     'T_g_loss': T_g_loss}

        del tape
        return S2T, T2S, S2T2S, loss_dict

    @tf.function
    def _train_segmentation(self, T, S2T, S_seg, T_g_loss):
        with tf.GradientTape() as tape:
            # segmentation for T
            T_latent1, T_latent2 = self.encoder(T, training=True)
            T_pred1 = self.segmentation1(T_latent1, training=True)
            T_pred2 = self.segmentation2(T_latent2, training=True)

            # segmentation for S2T
            S2T_latent1, S2T_latent2 = self.encoder(S2T, training=True)
            S2T_pred1 = self.segmentation1(S2T_latent1, training=True)
            S2T_pred2 = self.segmentation2(S2T_latent2, training=True)

            # segmentation discriminator
            T_pred1_d_logits = self.D_P(T_pred1, training=True)
            T_pred2_d_logits = self.D_P2(T_pred2, training=True)

            # gan loss (via discriminator)
            T_pred1_g_loss = generator_loss_lsgan(T_pred1_d_logits)
            T_pred2_g_loss = generator_loss_lsgan(T_pred2_d_logits)

            # task loss
            T_ce_loss = 0
            T_dice_loss = DiceLoss()(S_seg, S2T_pred1)
            T_dice_coeff = DiceCoefficient()(S_seg, S2T_pred1)
            T2_ce_loss = 0
            T2_dice_loss = DiceLoss()(S_seg, S2T_pred2)
            T2_dice_coeff = DiceCoefficient()(S_seg, S2T_pred2)

            # segmentation loss
            T_seg_loss = T_ce_loss + T_dice_loss + 0.1 * (
                    T2_ce_loss + T2_dice_loss) + 0.1 * T_g_loss + 0.1 * T_pred1_g_loss + 0.1 * T_pred2_g_loss

        segm_grad = tape.gradient(T_seg_loss,
                                  self.encoder.trainable_variables + self.segmentation1.trainable_variables + self.segmentation2.trainable_variables)
        self.segmentation_optimizer.apply_gradients(
            zip(segm_grad,
                self.encoder.trainable_variables + self.segmentation1.trainable_variables + self.segmentation2.trainable_variables))

        loss_dict = {'T_seg_loss': T_seg_loss,
                     'T_dice_loss': T_dice_loss,
                     'T2_dice_loss': T2_dice_loss,
                     'T_pred1_g_loss': T_pred1_g_loss,
                     'T_pred2_g_loss': T_pred2_g_loss,
                     'T_dice_coeff': T_dice_coeff,
                     'T2_dice_coeff': T2_dice_coeff}

        return T_pred1, T_pred2, S2T_pred1, S2T_pred2, loss_dict

    @tf.function
    def _train_discriminators(self, S, T, S2T_pool, T2S_pool, S2T2S):
        with tf.GradientTape(persistent=True) as tape:
            # discriminator of real samples
            S_d_logits, S_d_logits_aux = self.D_S(S, training=True)
            T_d_logits = self.D_T(T, training=True)
            S2T2S_d_logits, S2T2S_d_logits_aux = self.D_S(S2T2S, training=True)

            # discriminator of fake samples
            T2S_pool_d_logits, T2S_pool_d_logits_aux = self.D_S(T2S_pool, training=True)
            S2T_pool_d_logits = self.D_T(S2T_pool, training=True)

            # discriminator losses
            S_d_loss = discriminator_loss_lsgan(S_d_logits, T2S_pool_d_logits)
            S_aux_d_loss = discriminator_loss_lsgan(S2T2S_d_logits_aux, T2S_pool_d_logits_aux)
            S_d_loss = S_d_loss + S_aux_d_loss
            T_d_loss = discriminator_loss_lsgan(T_d_logits, S2T_pool_d_logits)

        D_S_grad = tape.gradient(S_d_loss, self.D_S.trainable_variables)
        D_T_grad = tape.gradient(T_d_loss, self.D_T.trainable_variables)

        self.D_S_optimizer.apply_gradients(
            zip(D_S_grad, self.D_S.trainable_variables))
        self.D_T_optimizer.apply_gradients(
            zip(D_T_grad, self.D_T.trainable_variables))
        del tape

        return {'S_d_loss': S_d_loss, 'T_d_loss': T_d_loss}

    @tf.function
    def _train_discriminator_segm(self, T_pred1, T_pred2, S2T_pred1, S2T_pred2):
        with tf.GradientTape(persistent=True) as tape:
            # discriminator predicated segm masks
            S2T_pred1_d_logits = self.D_P(S2T_pred1, training=True)
            T_pred1_d_logits = self.D_P(T_pred1, training=True)
            S2T_pred2_d_logits = self.D_P2(S2T_pred2, training=True)
            T_pred2_d_logits = self.D_P2(T_pred2, training=True)

            # discriminator segmentation losses
            P_d_loss = discriminator_loss_lsgan(T_pred1_d_logits, S2T_pred1_d_logits)
            P2_d_loss = discriminator_loss_lsgan(T_pred2_d_logits, S2T_pred2_d_logits)

        D_P_grad = tape.gradient(P_d_loss, self.D_P.trainable_variables)
        D_P2_grad = tape.gradient(P2_d_loss, self.D_P2.trainable_variables)
        self.D_P_optimizer.apply_gradients(
            zip(D_P_grad, self.D_P.trainable_variables))
        self.D_P2_optimizer.apply_gradients(
            zip(D_P2_grad, self.D_P2.trainable_variables))
        del tape

        return {'P_d_loss': P_d_loss, 'P2_d_loss': P2_d_loss}

    def train_step(self, S, T, S_seg, step, D_loss_dict):
        """
        One training step:
        1) train generators (generator and decoder)
        2) train segmentation (encoder + segmentation classifier)
        2) chose fake images from a pool (and fill pool with new generated images)
        3) train discriminator (for special number of steps, eg every 10th step, discriminator is trained)
        """

        # train generator and decoder
        S2T, T2S, S2T2S, G_loss_dict = self._train_generator_decoder(S, T)

        # train segmentation (encoder + decoder)
        T_pred1, T_pred2, S2T_pred1, S2T_pred2, Seg_loss_dict = self._train_segmentation(T, S2T, S_seg,
                                                                                         G_loss_dict[
                                                                                             "T_g_loss"])

        # chose T2S, S2T from history
        S2T_pool = self._fake_image_history(self._num_fake, S2T, self._fake_T)
        T2S_pool = self._fake_image_history(self._num_fake, T2S, self._fake_S)
        self._num_fake += 1

        # train discriminator for S and T
        if step % self.d_step == 0:
            discr_loss_dict = self._train_discriminators(S, T, S2T_pool, T2S_pool, S2T2S)

            # train discriminator segmentation
            discr_segm_loss_dict = self._train_discriminator_segm(T_pred1, T_pred2, S2T_pred1, S2T_pred2)

            D_loss_dict = {**discr_loss_dict, **discr_segm_loss_dict}

        return G_loss_dict, Seg_loss_dict, D_loss_dict

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
        logging.info("SIFA: set up summary file writer with directory {}.".format(directory))
        self.train_summary_writer = tf.summary.create_file_writer(directory)

    def _init_checkpoint(self, directory=os.path.join("tmp", 'checkpoints'), restore=True):
        """
        Initialize the model checkpoints.
        """
        logging.info("SIFA: set up checkpoints with directory {}.".format(directory))
        self.checkpoint = Checkpoint(dict(G_S2T=self.G_S2T,
                                          encoder=self.encoder,
                                          decoder=self.decoder,
                                          segmentation1=self.segmentation1,
                                          segmentation2=self.segmentation2,
                                          D_S=self.D_S,
                                          D_T=self.D_T,
                                          D_P=self.D_P,
                                          D_P2=self.D_P2,
                                          G_optimizer=self.G_S2T_optimizer,
                                          decoder_optimizer=self.decoder_optimizer,
                                          segmentation_optmizer=self.segmentation_optimizer,
                                          D_S_optimizer=self.D_S_optimizer,
                                          D_T_optimizer=self.D_T_optimizer,
                                          D_P_optimizer=self.D_P_optimizer,
                                          D_P2_optimizer=self.D_P2_optimizer),
                                     directory,
                                     max_to_keep=3)
        if restore:
            try:  # restore checkpoint including the epoch counter
                self.checkpoint.restore().assert_existing_objects_matched()
            except Exception as e:
                print("SIFA: " + e)

    @staticmethod
    def _collect_losses(G_loss_dict, D_loss_dict, S_loss_dict, G_loss_dict_list, D_loss_dict_list, S_loss_dict_list):
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
        for k, v in S_loss_dict.items():
            if type(S_loss_dict_list[k]) == list:
                S_loss_dict_list[k].append(v.numpy())
            else:
                S_loss_dict_list[k] = [v.numpy()]
        return G_loss_dict_list, D_loss_dict_list, S_loss_dict_list

    def _save_models(self):
        """
        Save models to SavedModel
        """
        logging.info("SIFA: save models to {}".format(self.dir_save_model))
        self.G_S2T.save(os.path.join(self.dir_save_model, "G_S2T"))
        self.encoder.save(os.path.join(self.dir_save_model, "Encoder"))
        self.decoder.save(os.path.join(self.dir_save_model, "Decoder"))
        self.segmentation1.save(os.path.join(self.dir_save_model, "Segmentation"))
        self.segmentation2.save(os.path.join(self.dir_save_model, "Segmentation2"))
        self.D_P.save(os.path.join(self.dir_save_model, "D_P"))
        self.D_P2.save(os.path.join(self.dir_save_model, "D_P2"))
        self.D_S.save(os.path.join(self.dir_save_model, "D_S"))
        self.D_T.save(os.path.join(self.dir_save_model, "D_T"))

    @tf.function
    def sample(self, S, T):
        """
        Generate samples from current generators.
        """
        S2T = self.G_S2T([S, S], training=False)
        S2T_lat, _ = self.encoder(S2T, training=True)
        S2T_pred = self.segmentation1(S2T_lat, training=False)

        T_lat, _ = self.encoder(T, training=False)
        T_pred = self.segmentation1(T_lat, training=False)

        return S2T, S2T_pred, T_pred

    def train(self, epochs=50, data_nr=None, restore=True, step_decay=None):
        """
        Train SIFA pipeline:
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
        logging.info("SIFA: set up training.")

        G_loss_dict_list = {k: 0 for k in ['S2T2S_c_loss', 'T2S2T_c_loss', 'S2T_g_loss', 'T2S_g_loss',
                                           'S_g_loss', 'T_g_loss']}
        Seg_loss_dict_list = {k: 0 for k in ['T_seg_loss', 'T_dice_loss', 'T2_dice_loss',
                                             'T_pred1_g_loss', 'T_pred2_g_loss',
                                             'T_dice_coeff', 'T2_dice_coeff']}
        D_loss_dict_list = {k: 0 for k in ['S_d_loss', 'T_d_loss', 'P_d_loss', 'P2_d_loss']}
        if data_nr is None or data_nr > len(self.train_set):
            data_nr = len(self.train_set)
        self._init_summary_file_writer(self.dir_tb)
        self._init_checkpoint(self.dir_cktp, restore=restore)
        self._set_optimizer()
        sample_counter = 0

        D_loss_dict = {'S_d_loss': tf.constant(100.0),
                       'T_d_loss': tf.constant(100.0)}

        logging.info("SIFA: start training.")
        for epoch in range(epochs + 1):
            total_time_per_epoch = 0
            print("Epoch {0}/{1}".format(epoch, epochs))
            for idx in range(data_nr):
                S, T_ = self.train_set[idx]
                S = S["image"]
                T = T_["t2"]
                S_seg = T_["vs"]
                start = time()
                G_loss_dict, Seg_loss_dict, D_loss_dict = self.train_step(S, T, S_seg, idx, D_loss_dict)
                elapsed = time() - start
                total_time_per_epoch += elapsed
                print(self.template.format(Seg_loss_dict["T_seg_loss"],
                                           Seg_loss_dict["T_dice_loss"],
                                           Seg_loss_dict["T_dice_coeff"],
                                           idx, data_nr - 1,
                                           total_time_per_epoch),
                      end="\r")
                self.train_set.reset()
                G_loss_dict_list, D_loss_dict_list, Seg_loss_dict_list = self._collect_losses(G_loss_dict,
                                                                                              D_loss_dict,
                                                                                              Seg_loss_dict,
                                                                                              G_loss_dict_list,
                                                                                              D_loss_dict_list,
                                                                                              Seg_loss_dict_list)
                if idx % self.sample_step == 0:
                    A, B_ = self.val_set[sample_counter]
                    A = A["image"]
                    B = B_["t2"]
                    T_seg = B_["vs_2"]
                    S2T, S2T_pred, T_pred = self.sample(A, B)
                    img = np.hstack(
                        np.concatenate([tf.expand_dims(A, -1), S2T, S2T_pred, tf.expand_dims(B, -1), T_pred,
                                        tf.expand_dims(T_seg, -1)],
                                       axis=0))
                    img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    cv2.imwrite(os.path.join(self.dir_sample, 'iter-%03d-%05d.jpg' % (epoch, idx)),
                                img)
                    sample_counter = sample_counter + 1
                    if sample_counter >= len(self.val_set):
                        sample_counter = 0

            with self.train_summary_writer.as_default():
                for k, v in G_loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                for k, v in Seg_loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                for k, v in D_loss_dict_list.items():
                    tf.summary.scalar(k, np.mean(v), step=epoch)
                tf.summary.scalar("learning_rate",
                                  self.G_S2T_optimizer.learning_rate,
                                  step=epoch)
            print(self.template.format(np.mean(Seg_loss_dict_list["T_seg_loss"]),
                                       np.mean(Seg_loss_dict_list["T_dice_loss"]),
                                       np.mean(Seg_loss_dict_list["T_dice_coeff"]),
                                       data_nr - 1, data_nr - 1, total_time_per_epoch))
            self.checkpoint.save(epoch)

        self._save_models()
