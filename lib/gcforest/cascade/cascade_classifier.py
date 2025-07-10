# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets.
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng.
"""
import numpy as np
import os
import os.path as osp
import pickle

from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb

LOGGER = get_logger('gcforest.cascade.cascade_classifier')


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


class CascadeClassifier(object):
    def __init__(self, ca_config):
        """
        Parameters (ca_config)
        ----------
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in early_stopping_rounds, the cascade level will stop automatically growing
        max_layers: int
            maximum number of cascade layers allowed for exepriments, 0 means use Early Stoping to automatically find the layer number
        n_classes: int
            Number of classes
        est_configs:
            List of CVEstimator's config
        look_indexs_cycle (list 2d): default=None
            specification for layer i, look for the array in look_indexs_cycle[i % len(look_indexs_cycle)]
            defalut = None <=> [range(n_groups)]
            .e.g.
                look_indexs_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3; layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermidiate results in data_save_dir
            if data_save_rounds = 0, then no savings for intermidiate results
        """
        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        # self.data_save_dir = self.get_value("data_save_dir", None, basestring)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.opt_layer_num = -1
        # LOGGER.info("\n" + json.dumps(ca_config, sort_keys=True, indent=4, separators=(',', ':')))

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="cascade")

    def _set_estimator(self, li, ei, est):
        if li not in self.estimator2d:
            self.estimator2d[li] = {}
        self.estimator2d[li][ei] = est

    def _get_estimator(self, li, ei):
        return self.estimator2d.get(li, {}).get(ei, None)

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # n_classes
        if "n_classes" in est_args:
            est_args["n_classes"] = int(est_args["n_classes"])        
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def _check_look_indexs_cycle(self, X_groups, is_fit):
        # check look_indexs_cycle
        n_groups = len(X_groups)
        if is_fit and self.look_indexs_cycle is None:
            look_indexs_cycle = [list(range(n_groups))]
        else:
            look_indexs_cycle = self.look_indexs_cycle
            for look_indexs in look_indexs_cycle:
                if np.max(look_indexs) >= n_groups or np.min(look_indexs) < 0 or len(look_indexs) == 0:
                    raise ValueError("look_indexs doesn't match n_groups!!! look_indexs={}, n_groups={}".format(
                        look_indexs, n_groups))
        if is_fit:
            self.look_indexs_cycle = look_indexs_cycle
        return look_indexs_cycle

    def _check_group_dims(self, X_groups, is_fit):
        if is_fit:
            group_starts, group_ends, group_dims = [], [], []
        else:
            group_starts, group_ends, group_dims = self.group_starts, self.group_ends, self.group_dims
        n_datas = X_groups[0].shape[0]
        X = np.zeros((n_datas, 0), dtype=X_groups[0].dtype)
        for i, X_group in enumerate(X_groups):
            assert(X_group.shape[0] == n_datas)
            X_group = X_group.reshape(n_datas, -1)
            if is_fit:
                group_dims.append( X_group.shape[1] )
                group_starts.append(0 if i == 0 else group_ends[i - 1])
                group_ends.append(group_starts[i] + group_dims[i])
            else:
                assert(X_group.shape[1] == group_dims[i])
            X = np.hstack((X, X_group))
        if is_fit:
            self.group_starts, self.group_ends, self.group_dims = group_starts, group_ends, group_dims
        return group_starts, group_ends, group_dims, X

    def fit_transform(self, X_groups_train, y_train, X_groups_test, y_test, stop_by_test=False, train_config=None, dX=None, py=None):
        """
        fit until the accuracy converges in early_stop_rounds
        """
        if train_config is None:
            from ..config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir

        is_eval_test = "test" in train_config.phases
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        if is_eval_test and not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_train.shape={},y_train.shape={},X_groups_test.shape={},y_test.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape,
            [xt.shape for xt in X_groups_test] if is_eval_test else "no_test", y_test.shape if is_eval_test else "no_test"))

        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_train, True)
        if is_eval_test:
            self._check_look_indexs_cycle(X_groups_test, False)

        group_starts, group_ends, group_dims, X_train = self._check_group_dims(X_groups_train, True)
        if is_eval_test:
            _, _, _, X_test = self._check_group_dims(X_groups_test, False)
        else:
            X_test = np.zeros((0, X_train.shape[1]))
        
        if dX is None:
            dX = np.zeros_like(X_train)
        elif dX.shape != X_train.shape:
            raise ValueError("The shape of dX must match the flattened shape of X_train")

        n_trains = X_groups_train[0].shape[0]
        n_tests = X_groups_test[0].shape[0] if is_eval_test else 0
        n_classes = self.n_classes
        train_acc_list, test_acc_list = [], []
        opt_datas = [None, None, None, None]

        try:
            # Initialize arrays for augmented features and their uncertainties
            X_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            dX_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            X_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
            dX_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
            
            layer_id = 0
            while 1:
                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                
                # --- Construct input for the current layer ---
                if layer_id == 0:
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                    dX_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                    if n_tests > 0:
                        X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                        dX_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                else:
                    X_cur_train = X_proba_train.copy()
                    dX_cur_train = dX_proba_train.copy()
                    if n_tests > 0:
                        X_cur_test = X_proba_test.copy()
                        dX_cur_test = dX_proba_test.copy()
                
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
                for i in look_indexs:
                    X_cur_train = np.hstack((X_cur_train, X_train[:, group_starts[i]:group_ends[i]]))
                    dX_cur_train = np.hstack((dX_cur_train, dX[:, group_starts[i]:group_ends[i]]))
                    if n_tests > 0:
                        X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
                        # For simplicity, test uncertainty is zero for original features
                        dX_cur_test = np.hstack((dX_cur_test, np.zeros_like(X_test[:, group_starts[i]:group_ends[i]])))
                
                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}".format(layer_id, look_indexs, X_cur_train.shape))
                
                y_train_proba_li = np.zeros((n_trains, n_classes))
                y_test_proba_li = np.zeros((n_tests, n_classes))
                
                for ei, est_config in enumerate(self.est_configs):
                    est = self._init_estimators(layer_id, ei)
                    
                    test_sets = [("test", X_cur_test, y_test, {"dX": dX_cur_test})] if n_tests > 0 else None
                    
                    fit_args = {"dX": dX_cur_train, "py": py}
                    y_probas, y_probas_dX = est.fit_transform(
                        X_cur_train, y_train, y_train,
                        test_sets=test_sets, eval_metrics=self.eval_metrics,
                        keep_model_in_mem=train_config.keep_model_in_mem,
                        **fit_args
                    )
                    
                    # Store mean predictions
                    X_proba_train[:, ei * n_classes: (ei + 1) * n_classes] = y_probas[0]
                    y_train_proba_li += y_probas[0]
                    
                    # Store uncertainty predictions
                    dX_proba_train[:, ei * n_classes: (ei + 1) * n_classes] = y_probas_dX[0]

                    if n_tests > 0:
                        X_proba_test[:, ei * n_classes: (ei + 1) * n_classes] = y_probas[1]
                        dX_proba_test[:, ei * n_classes: (ei + 1) * n_classes] = y_probas_dX[1]
                        y_test_proba_li += y_probas[1]
                    
                    if train_config.keep_model_in_mem:
                        self._set_estimator(layer_id, ei, est)
                
                y_train_proba_li /= len(self.est_configs)
                train_avg_acc = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1), 'layer_{} - train.classifier_average'.format(layer_id))
                train_acc_list.append(train_avg_acc)
                
                if n_tests > 0:
                    y_test_proba_li /= len(self.est_configs)
                    test_avg_acc = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1), 'layer_{} - test.classifier_average'.format(layer_id))
                    test_acc_list.append(test_avg_acc)
                else:
                    test_acc_list.append(0.0)

                opt_layer_id = get_opt_layer_id(test_acc_list if stop_by_test else train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [X_proba_train, y_train, X_proba_test if n_tests > 0 else None, y_test]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                    # log and save final result (opt layer)
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                        opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
                    if data_save_dir is not None:
                        self.save_data( data_save_dir, opt_layer_id, *opt_datas)
                    # remove unused model
                    if train_config.keep_model_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            for ei, est_config in enumerate(self.est_configs):
                                self._set_estimator(li, ei, None)
                    self.opt_layer_num = opt_layer_id + 1
                    return opt_layer_id, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(data_save_dir, layer_id, *opt_datas)
                # inc layer_id
                layer_id += 1
            LOGGER.info("[Result][Reach Max Layer] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
            if data_save_dir is not None:
                self.save_data(data_save_dir, self.max_layers - 1, *opt_datas)
            self.opt_layer_num = self.max_layers
            return self.max_layers, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
        except KeyboardInterrupt:
            pass

    def transform(self, X_groups_test, dX=None):
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_test.shape={}".format([xt.shape for xt in X_groups_test]))
        
        # Flatten the input feature groups into a single X_test array
        group_starts, group_ends, group_dims, X_test = self._check_group_dims(X_groups_test, False)
        
        # Handle the corresponding dX array
        if dX is None:
            dX = np.zeros_like(X_test)
        elif dX.shape != X_test.shape:
             raise ValueError("The shape of dX must match the flattened shape of X_test during transform")

        LOGGER.info("X_test.shape={}".format(X_test.shape))

        n_tests = X_groups_test[0].shape[0]
        n_classes = self.n_classes

        # Initialize arrays to store augmented features and their uncertainties
        X_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
        dX_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)

        for layer_id in range(self.opt_layer_num):
            
            # Construct input for the current layer
            if layer_id == 0:
                X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                dX_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
            else:
                X_cur_test = X_proba_test.copy()
                dX_cur_test = dX_proba_test.copy()
            
            look_indexs = self.look_indexs_cycle[layer_id % len(self.look_indexs_cycle)]
            for i in look_indexs:
                X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
                dX_cur_test = np.hstack((dX_cur_test, dX[:, group_starts[i]:group_ends[i]]))
            
            LOGGER.info("[layer={}] look_indexs={}, X_cur_test.shape={}".format(layer_id, look_indexs, X_cur_test.shape))
            
            # Make predictions with the estimators from this layer
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                if est is None:
                    raise ValueError(f"Model for layer {layer_id}, estimator {ei} not found. Ensure keep_model_in_mem=True during training.")
                
                predict_args = {"dX": dX_cur_test}
                
                # Check for the uncertainty-aware prediction method
                if hasattr(est, "predict_proba_with_std"):
                    mean_proba, std_proba = est.predict_proba_with_std(X_cur_test, **predict_args)
                else:
                    mean_proba = est.predict_proba(X_cur_test)
                    std_proba = np.zeros_like(mean_proba)

                # Store results to be used as input for the next layer
                X_proba_test[:, ei * n_classes:(ei + 1) * n_classes] = mean_proba
                dX_proba_test[:, ei * n_classes:(ei + 1) * n_classes] = std_proba
                
        # Return the augmented features from the FINAL layer
        return X_proba_test

    def predict_proba(self, X, dX=None):
        y_proba = self.transform(X, dX=dX)
        
        # Reshape and average the predictions from all estimators in the final layer
        y_proba = y_proba.reshape((y_proba.shape[0], self.n_estimators_1, self.n_classes))
        y_proba = y_proba.mean(axis=1)
        
        return y_proba
    
    def save_data(self, data_save_dir, layer_id, X_train, y_train, X_test, y_test):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train} if pi == 0 else {"X": X_test, "y": y_test}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
