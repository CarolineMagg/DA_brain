########################################################################################################################
# Simple supervised Segmentation pipeline
########################################################################################################################
import tensorflow as tf
from losses.dice import DiceLoss, DiceCoefficient

__author__ = "c.magg"


class SimpleSegmentation:

    def __init__(self, identifier="test"):
        # dataset
        self.train_set = None
        self.val_set = None
        self.test_set = None

        # callbacks
        self.callbacks = []
        self.identifier = identifier
        self.save_path = "../saved_models/" + identifier + ".hdf5"
        self.set_callbacks(identifier)

        # model
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.losses = DiceLoss()
        self.metrics = DiceCoefficient()

    def set_callbacks(self, identifier):
        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=15, verbose=1,
                                                     mode='min')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.save_path, monitor="val_loss", verbose=1, save_best_only=True,
                                                        save_weights_only=False)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir="../logs/" + identifier)

        self.callbacks = [earlystop,
                          checkpoint,
                          tensorboard]

    def set_model(self, model):
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=self.losses, metrics=self.metrics)

    def set_data(self, train, val, test):
        self.train_set = train
        self.val_set = val
        self.test_set = test

    def check_for_training(self):
        if self.train_set is None or self.test_set is None or self.val_set is None:
            raise ValueError("Set train/val/test datasets.")
        if self.model is None:
            raise ValueError("Set model.")
        if len(self.callbacks) == 0:
            raise ValueError("Set callbacks.")

    def fit(self, init_epoch=0, epochs=2):
        print("Training ....")
        self.model.fit(self.train_set,
                       validation_data=self.val_set,
                       callbacks=self.callbacks,
                       initial_epoch=init_epoch,
                       epochs=epochs,
                       verbose=1)

    def evaluate(self):
        print("Evaluation ....")
        self.model.load_weights(self.save_path)
        self.model.evaluate(self.val_set)
        self.model.evaluate(self.test_set)
