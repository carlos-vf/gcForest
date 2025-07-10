import numpy as np

from .cascade.cascade_classifier import CascadeClassifier
from .config import GCTrainConfig
from .fgnet import FGNet
from .utils.log_utils import get_logger

LOGGER = get_logger("gcforest.gcforest")


class GCForest(object):
    def __init__(self, config):
        self.config = config
        self.train_config = GCTrainConfig(config.get("train", {}))
        if "net" in self.config:
            self.fg = FGNet(self.config["net"], self.train_config.data_cache)
        else:
            self.fg = None
        if "cascade" in self.config:
            self.ca = CascadeClassifier(self.config["cascade"])
        else:
            self.ca = None

    def fit_transform(self, X_train, y_train=None, X_test=None, y_test=None, train_config=None, dX=None, py=None):
        """
        Fit the GCForest model.

        If `y_train` is not provided, discrete labels will be derived from `py`.
        """
        if y_train is None:
            if py is None:
                raise ValueError(
                    "Either 'y_train' (discrete labels) or 'py' (probabilistic labels) "
                    "must be provided to the fit_transform method."
                )
            LOGGER.info("y_train not provided. Deriving discrete labels from py using argmax.")
            y_train = np.argmax(py, axis=1)

        train_config = train_config or self.train_config
        if X_test is None or y_test is None:
            if "test" in train_config.phases:
                train_config.phases.remove("test")
            X_test, y_test = None, None
        
        if self.fg is not None:
            self.fg.fit_transform(X_train, y_train, X_test, y_test, train_config)
            X_train = self.fg.get_outputs("train")
            if "test" in train_config.phases:
                X_test = self.fg.get_outputs("test")
        
        if self.ca is not None:
            _, X_train, _, X_test, _, = self.ca.fit_transform(
                X_train, y_train, X_test, y_test, train_config=train_config, dX=dX, py=py
            )

        if X_test is None:
            return X_train
        else:
            return X_train, X_test

    def transform(self, X, dX=None):
        if self.fg is not None:
            X = self.fg.transform(X)
        y_proba = self.ca.transform(X, dX=dX)
        return y_proba

    def predict_proba(self, X, dX=None):
        if self.fg is not None:
            X = self.fg.transform(X)
        y_proba = self.ca.predict_proba(X, dX=dX)
        return y_proba

    def predict(self, X, dX=None):
        y_proba = self.predict_proba(X, dX=dX)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred


    def set_data_cache_dir(self, path):
        self.train_config.data_cache.cache_dir = path

    def set_keep_data_in_mem(self, flag):
        """
        flag (bool):
            if flag is 0, data will not be keeped in memory.
            this is for the situation when memory is the bottleneck
        """
        self.train_config.data_cache.config["keep_in_mem"]["default"] = flag

    def set_keep_model_in_mem(self, flag):
        """
        flag (bool):
            if flag is 0, model will not be keeped in memory.
            this is for the situation when memory is the bottleneck
        """
        self.train_config.keep_model_in_mem = flag
