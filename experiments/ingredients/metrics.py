import torch
from ignite.metrics import (
    Accuracy,
    ConfusionMatrix,
    EpochMetric,
    Fbeta,
    Loss,
    Precision,
    RunningAverage,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.functional import one_hot

IGNITE_METRIC_NAMES = [
    "loss",
    "acc",
    "auc",
    "precision",
    "f1",
    "cm",
    "auprc",
]


class TemporalLossWrapper(nn.Module):
    """Applies a given loss_str function along the length of the outputs removing any nans in the given labels_split.

    Given a PyTorch loss_str function and target labels_split of shape [N, L, C] (as opposed to [N, C]), this removes
    any values where labels_split are nan (corresponding to a finished time series) and computes the loss_str against
    the labels_split and predictions in the remaining non nan locations.
    """

    def __init__(self, criterion):
        """
        Args:
            criterion (nn.Module): A pytorch loss_str function.
        """
        super().__init__()

        assert isinstance(criterion, nn.Module)
        self.criterion = criterion

    def forward(self, preds, labels):
        mask = ~torch.isnan(labels)
        return self.criterion(preds[mask], labels[mask])


class RMSELoss(nn.Module):
    """ Root mean squere"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


def setup_ignite_metrics(metric_names, loss_fn, wrap_temporal=False):
    """Setup the metrics given metric name strings.

    Args:
        metric_names (list of strings): A list of strings specifying the names of the metrics. Each string must reside
            in IGNITE_METRICS.keys().
        loss_fn (nn.Module): A torch loss_str function from IGNITE_LOSSES.
        wrap_temporal (bool): Set True to wrap the losses in `TemporalLossWrapper`.

    Returns:
        Two dictionaries, these provide the training and validation metrics.
    """
    allowed_metrics = IGNITE_METRIC_NAMES
    for metric_name in metric_names:
        if metric_name not in allowed_metrics:
            raise NotImplementedError(
                "Allowed metrics are {}, recieved {}.".format(
                    allowed_metrics, metric_name
                )
            )

    # Get the output transforms
    train_probas, train_preds, val_probas, val_preds = ready_output_transforms(loss_fn)

    # Setup the metrics
    train_metrics = {
        "loss": RunningAverage(output_transform=lambda x: x[0]),
        "acc": Accuracy(output_transform=train_preds),
        "auc": IgniteAUC(output_transform=train_probas),
        "auprc": IgniteAUPRC(output_transform=train_probas),
        "precision": Precision(output_transform=train_preds),
        "f1": Fbeta(1, output_transform=train_preds),
    }
    val_metrics = {
        "loss": Loss(loss_fn),
        "acc": Accuracy(output_transform=val_preds),
        "auc": IgniteAUC(output_transform=val_probas),
        "auprc": IgniteAUPRC(output_transform=val_probas),
        "precision": Precision(output_transform=val_preds),
        "f1": Fbeta(1, output_transform=val_preds),
    }

    # Subset the metrics
    train_metrics = {name: train_metrics[name] for name in metric_names}
    val_metrics = {name: val_metrics[name] for name in metric_names}

    return train_metrics, val_metrics


def ready_output_transforms(loss_fn):
    """Readies the output transforms lambda functions.

    These differ if the output is binary hence the need to call with `loss_fn`.
    """
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):

        def prediction_transform(x):
            return torch.round(torch.sigmoid(x))

    else:

        def prediction_transform(x):
            return x

    def train_probas(x):
        mask = ~torch.isnan(x[1])
        return (torch.sigmoid(x[2][mask]), x[1][mask])

    def val_probas(x):
        mask = ~torch.isnan(x[1])
        return (torch.sigmoid(x[0][mask]), x[1][mask])

    def train_preds(x):
        mask = ~torch.isnan(x[1])
        return (prediction_transform(x[2][mask]), x[1][mask])

    def val_preds(x):
        mask = ~torch.isnan(x[1])
        return (prediction_transform(x[0][mask]), x[1][mask])

    return train_probas, train_preds, val_probas, val_preds


def ignite_binary_confusion_matrix(output_transform, num_classes=2):
    """Confusion matrix hack for ignite.

    Args:
        output_transform (function): This must output y_pred, y as normal.
        num_classes (int): Number of output classes.

    Returns:
        A converted output transform that can be used in ConfusionMatrix.
    """

    def converter(x):
        y_pred, y = output_transform(x)
        y_ohe = one_hot(y_pred.to(torch.int64), num_classes=num_classes).view(
            -1, num_classes
        )
        return y_ohe, y.view(-1).to(torch.int64)

    cm = ConfusionMatrix(num_classes=num_classes, output_transform=converter)
    return cm


def roc_auc_compute_fn(y_preds, y_targets):
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return roc_auc_score(y_true, y_pred)


def auprc_compute_fn(y_preds, y_targets):
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return average_precision_score(y_true, y_pred)


class IgniteAUC(EpochMetric):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    accumulating predictions and the ground-truth during an epoch and applying
    `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ .

    Args:
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        check_compute_fn (bool): Optional default False. If True, `roc_curve
            <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#
            sklearn.metrics.roc_auc_score>`_ is run on the first batch of data to ensure there are
            no issues. User will be warned in case there are any issues computing the function.

    ROC_AUC expects y to be comprised of 0's and 1's. y_pred must either be probability estimates or confidence
    values. To apply an activation to y_pred, use output_transform as shown below:

    .. code-block:: python

        def activated_output_transform(output):
            y_pred, y = output
            y_pred = torch.sigmoid(y_pred)
            return y_pred, y

        roc_auc = ROC_AUC(activated_output_transform)

    """

    def __init__(self, output_transform=lambda x: x, check_compute_fn: bool = False):
        super(IgniteAUC, self).__init__(
            roc_auc_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )


class IgniteAUPRC(EpochMetric):
    """ See documentation for Ignite AUROC. """

    def __init__(self, output_transform=lambda x: x, check_compute_fn: bool = False):
        super(IgniteAUPRC, self).__init__(
            auprc_compute_fn,
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )
