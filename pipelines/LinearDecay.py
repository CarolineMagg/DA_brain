########################################################################################################################
# LinearDecay for LR taken from
# taken from https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/1aa6398d918875a1cf25320881d09b7d9f3f63b8/module.py#L115
########################################################################################################################

import tensorflow as tf


class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    LR Scheduler for linear decay:
        if `step` < `step_decay`: use fixed learning rate
        else: linearly decay the learning rate to zero
    """

    def __init__(self, initial_learning_rate, total_steps, step_decay, **kwargs):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: tf.math.maximum(self._initial_learning_rate * (
                        1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)), 0),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate

    def get_config(self):
        return {"learning_rate": self.current_learning_rate, "initial_learning_rate": self._initial_learning_rate,
                "total_steps": self._total_steps, "step_decay": self._step_decay}
