import numpy as np


class EarlyStopping:
    """
    Keeps track of when the loss does not improve after a given patience.
    Useful to stop training when the validation loss does not improve anymore.
    Modified from https://github.com/azizilab/decipher/blob/main/decipher/tools/utils.py#L4.

    Args:
        patience: How long to wait after the last validation loss improvement.

    Returns:
        True if the training should stop.
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.validation_loss_min = np.inf

    def __call__(self, validation_loss):
        """Returns True if the training should stop."""
        if validation_loss < self.validation_loss_min:
            self.validation_loss_min = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def has_stopped(self):
        """Returns True if the stopping condition has been met."""
        return self.early_stop