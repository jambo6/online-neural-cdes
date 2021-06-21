"""
trainer.py
==========================
Generic model training as a sacred ingredient.
"""
import logging

# flake8: noqa
import os
import tempfile
import time
from collections import OrderedDict

import numpy as np
import torch
from ignite.engine import Events
from ignite.engine import _prepare_batch as ignite_prepare_batch
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ingredients.metrics import TemporalLossWrapper, setup_ignite_metrics
from sacred import Ingredient
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .metrics import RMSELoss

train_ingredient = Ingredient("trainer")


def set_gpu_idx():
    """GPU index is required to be stored in an environment variable (for parallelisation reasons)."""
    gpu_idx = os.environ.get("GPU")
    if gpu_idx is None:
        logging.warning("GPU not found in environment variable, setting manually as -1")
        gpu_idx = -1
    else:
        gpu_idx = int(gpu_idx)
    return gpu_idx


@train_ingredient.config
def train_config():
    loss_str = None
    wrap_temporal_loss = True
    optimizer_name = "adam"
    lr = None
    max_epochs = 1000
    metrics = ["loss", "acc"]
    prepare_batch = None
    val_metric_to_monitor = "loss"
    print_freq = 5
    epoch_per_metric = 1
    plateau_patience = 15
    plateau_terminate = 60
    gpu_if_available = True
    gpu_idx = set_gpu_idx()
    return_memory = True
    save_dir = None
    verbose = 0
    save_results_to_run = True
    evaluate_on_test = None
    final_metrics = None


@train_ingredient.capture
def train(
    _run,
    model,
    train_dl,
    val_dl,
    test_dl,
    loss_str=None,
    wrap_temporal_loss=None,
    optimizer_name=None,
    lr=None,
    max_epochs=None,
    metrics=None,
    prepare_batch=None,
    val_metric_to_monitor=None,
    print_freq=None,
    epoch_per_metric=None,
    plateau_patience=None,
    plateau_terminate=None,
    gpu_if_available=True,
    gpu_idx=None,
    return_memory=None,
    save_results_to_run=None,
    evaluate_on_test=None,
    verbose=None,
):
    """Model training function for pytorch networks using the ignite module.

    This builds and runs a standard training process using the ignite framework. Given train/val/test dataloaders,
    attaches specified metrics and runs over the training set with LR scheduling, early stopping, and model
    check-pointing all built in.

    Args:
        _run (sacred object): The sacred run object.
        model (nn.Module): A pytorch module.
        train_dl (DataLoader): Training dataloader.
        val_dl (DataLoader): Validation dataloader.
        test_dl (DataLoader): Test dataloader.
        optimizer_name (str): Name of the optimizer to use. Accepts ('adam', 'sgd').
        lr (float or None): The initial value of the learning rate. If left as none will use (0.01 * 32 / batch_size).
        loss_str (function): The loss function. Can be a string in ('ce', 'bce', 'mse') or a torch.nn.Module loss.
        wrap_temporal_loss (bool): Set True to wrap in a temporal loss class. This is used when the labels_split are
            classification but the problem is online.
        max_epochs (int): Max epochs to run the algorithm for.
        metrics (list): A list of metric strings to be monitored.
        val_metric_to_monitor (str): The metric to monitor for LR scheduling and early stopping.
        print_freq (int): Frequency of printing train/val results to console.
        epoch_per_metric (int): Number of epochs before next computation of val metrics.
        plateau_patience (int): Number of epochs with no improvement before LR reduction.
        plateau_terminate (int): Number of epochs with no improvement before stopping.
        gpu_if_available (bool): Run on the gpu if one exists.
        gpu_idx (int): The index of the gpu to run on.
        return_memory (bool): Set True to attempt to get the memory stats for the run. This seems to fail on newer
            versions of pytorch so beware.
        save_results_to_run (bool): Set True to save all metrics to the run.
        evaluate_on_test (bool): Set False to skip test evaluation (saves a bit of time when hyperopting).
        verbose (int): Set 0 for no output 1 for all output.

    Returns:
        Trained model, dictionary of results, and a dictionary containing the full training history.
    """
    # Get device information
    device = set_device(gpu_if_available, gpu_idx=gpu_idx, model=model)

    # Extract the loss_str function
    loss_fn = set_loss(loss_str, wrap_temporal_loss=wrap_temporal_loss)

    # Specify a learning rate
    lr = set_lr(train_dl) if lr is None else lr

    # Ready the optimizer
    optimizer = setup_optimizer(model, optimizer_name, lr)

    # Choose metrics given the string list
    train_metrics, val_metrics = setup_ignite_metrics(
        metrics, loss_fn, wrap_temporal=wrap_temporal_loss
    )

    # Build engines
    trainer, evaluator, tester = build_engines(
        model,
        optimizer,
        loss_fn,
        device,
        train_metrics,
        val_metrics,
        prepare_batch=prepare_batch,
    )

    # Attach a validation logging function to the trainer
    validation_history, pbar = ready_validation_logging(
        trainer,
        evaluator,
        val_dl,
        train_metrics,
        val_metrics,
        epoch_per_metric,
        print_freq,
        max_epochs,
        verbose,
        test_dl=test_dl,
        tester=tester,
    )

    # Setup scheduling, early stopping and checkpointing
    _, _, checkpoint, checkpoint_tempdir, score_sign = add_handlers(
        model,
        trainer,
        evaluator,
        optimizer,
        val_metric_to_monitor,
        plateau_patience,
        plateau_terminate,
    )

    # Train the model
    elapsed, memory_usage, nfe = _train(
        model, trainer, train_dl, max_epochs, device, return_memory=return_memory
    )

    # Score on test
    model.load_state_dict(torch.load(checkpoint.last_checkpoint))
    evaluator.run(test_dl, max_epochs=1)

    # Compile a results dictionary
    results = compile_results(
        model,
        evaluator,
        validation_history,
        lr,
        elapsed,
        memory_usage,
        nfe,
        score_sign,
        val_metric_to_monitor,
    )

    # Process and print. Some processing is needed if ConfusionMatrix is used as a metric
    print_val_results(history=results, epoch=None, pbar=pbar, verbose=verbose)

    # Save results to _run
    if save_results_to_run:
        save_results(_run, results, validation_history)

    # Close the checkpoint temp file to remove it
    checkpoint_tempdir.cleanup()

    return model, results, validation_history


def get_freest_gpu():
    """GPU with most available memory."""
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def set_device(gpu_if_available, gpu_idx=None, model=None):
    """Sets the cuda device to run on.

    This will only set if `gpu_is_available` is marked True. If `gpu_idx` is set, then will run on that gpu index, else
    will find the free-est gpu in terms of available memory and run on that.

    Args:
        gpu_if_available (bool): Set True to allow a gpu to be selected.
        gpu_idx (int): The index of the GPU to run on.
        model (nn.Module): If given will run model.to(device).

    Returns:
        torch.device: The GPU device.
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if gpu_if_available:
            if gpu_idx > -1:
                device = torch.device("cuda:{}".format(gpu_idx))
                torch.cuda.set_device(gpu_idx)
            else:
                gpu = get_freest_gpu()
                device = torch.device("cuda:{}".format(gpu))
                torch.cuda.set_device(device)
    if model is not None:
        model.to(device)
    return device


def set_loss(loss_str, wrap_temporal_loss=False):
    """Gets the loss_str module if it exists."""
    IGNITE_LOSSES = {
        "ce": nn.CrossEntropyLoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "mse": nn.MSELoss(),
        "rmse": RMSELoss(),
    }

    loss = IGNITE_LOSSES.get(loss_str)
    if loss_str not in IGNITE_LOSSES.keys():
        raise NotImplementedError(
            "Allowed losses {}, got {}.".format(IGNITE_LOSSES.keys(), loss_str)
        )
    if wrap_temporal_loss:
        loss = TemporalLossWrapper(loss)

    return loss


def setup_optimizer(model, optimizer_name, lr, final_layer_scaling=10):
    """Sets up the optimizer according to a given name.

    If the model is a NeuralRDE, will multiply the learning rate of the final layer.

    Args:
        model (nn.Module): A PyTorch model.
        optimizer_name (str): Name from ['adam', 'sgd']
        lr (float): The main value of the lr.
        final_layer_scaling (float): Final layer lr becomes `lr * final_layer_scaling`.

    Returns:
        optimizer: A PyTorch optimizer.
    """
    optimizers = {"adam": optim.Adam, "sgd": optim.SGD}

    # Any param that has name starting 'final_linear' gets a multiplied lr by `final_layer_scaling`
    ps = []
    for name, param in model.named_parameters():
        lr_ = lr if not name.startswith("final_linear") else lr * final_layer_scaling
        ps.append({"params": param, "lr": lr_})
    optimizer = optimizers.get(optimizer_name)

    if optimizer is None:
        raise NotImplementedError(
            "Allowed optimizers {}, got {}.".format(optimizers.keys(), optimizer_name)
        )
    else:
        optimizer = optimizer(ps)

    return optimizer


def set_lr(train_dl):
    """Set as 0.01"""
    # batch_size = train_dl.batch_sampler.batch_size if hasattr(train_dl, 'batch_sampler') else train_dl.batch_size
    return 5e-3


def build_engines(
    model, optimizer, loss_fn, device, train_metrics, val_metrics, prepare_batch=None
):
    """Build the supervised trainers with ignite."""
    # Trainer
    def trainer_output_transform(x, y, y_pred, loss):
        return (loss.item(), y, y_pred)

    if prepare_batch is None:
        prepare_batch = ignite_prepare_batch

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        output_transform=trainer_output_transform,
        prepare_batch=prepare_batch,
    )

    # Evaluator
    evaluator = create_supervised_evaluator(
        model, device=device, metrics=val_metrics, prepare_batch=prepare_batch
    )

    # Tester
    tester = create_supervised_evaluator(
        model, device=device, metrics=val_metrics, prepare_batch=prepare_batch
    )

    # Attach metrics to trainer
    for name, metric in train_metrics.items():
        metric.attach(trainer, name)

    return trainer, evaluator, tester


def ready_validation_logging(
    trainer,
    evaluator,
    val_dl,
    train_metrics,
    val_metrics,
    epoch_per_metric,
    print_freq,
    max_epochs,
    verbose,
    test_dl=None,
    tester=None,
):
    """Attaches a function to the trainer to log the validaiton and training metrics after a number of epochs."""
    # tqdm progress bar
    class FakePbar:
        def update(self, x):
            pass

        def write(self, x):
            pass

    pbar = (
        tqdm(range(max_epochs), desc="Model training status", leave=True)
        if verbose > 0
        else FakePbar()
    )

    # For storing the validation history
    validation_history = OrderedDict()
    for key in train_metrics.keys():
        validation_history["{}.train".format(key)] = []
    for key in val_metrics.keys():
        validation_history["{}.val".format(key)] = []

    # Validation loop
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(engine):
        epoch = engine.state.epoch
        pbar.update(1)

        if (epoch % epoch_per_metric == 0) or (epoch == 0):
            add_metrics_to_dict(trainer.state.metrics, validation_history, ".train")

            evaluator.run(val_dl, max_epochs=1)
            add_metrics_to_dict(evaluator.state.metrics, validation_history, ".val")

            if verbose > 0:
                if (epoch % print_freq == 0) or (epoch == 0):
                    print_val_results(validation_history, epoch=epoch, pbar=pbar)
                    # print('\n--------------- TEST RESULTS ---------------- ')
                    # tester.run(test_dl, max_epochs=1)
                    # print(tester.state.metrics.items())
                    # print('--------------- TEST RESULTS ---------------- ')

    return validation_history, pbar


def add_handlers(
    model,
    trainer,
    evaluator,
    optimizer,
    val_metric_to_monitor,
    plateau_patience,
    plateau_terminate,
):
    """Adds lr reduction, early stopping, and checkpointing handlers."""
    # Score to monitor for early stopping and check-pointing
    sign = -1 if val_metric_to_monitor == "loss" else 1

    def score_function(engine):
        return engine.state.metrics[val_metric_to_monitor] * sign

    # LR scheduling (monitors validation loss), early stopping and check-pointing
    scheduler = ReduceLROnPlateau(
        optimizer, patience=plateau_patience, threshold=1e-6, min_lr=1e-7
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda engine: scheduler.step(engine.state.metrics["loss"]),
    )

    # Early stopping
    stopping = EarlyStopping(
        patience=plateau_terminate, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopping)

    # Checkpoint
    tempdir = tempfile.TemporaryDirectory()
    checkpoint = ModelCheckpoint(tempdir.name, "", score_function=score_function)
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpoint, {"best_model": model}
    )

    return scheduler, stopping, checkpoint, tempdir, sign


def _train(model, trainer, train_dl, max_epochs, device, return_memory=True):
    """Gets statistics on the time and memory stats."""
    # Time and memory if specified
    start_time = time.time()
    memory_usage = False
    start_memory = False
    if return_memory:
        start_memory = get_memory(device, reset=True)

    # Train
    trainer.run(train_dl, max_epochs=max_epochs)

    # Get time and memory stats
    elapsed = time.time() - start_time
    if start_memory:
        memory_usage = get_memory(device) - start_memory

    # NFEs
    nfe = None
    if hasattr(model, "nfe"):
        nfe = model.nfe

    return elapsed, memory_usage, nfe


def compile_results(
    model,
    evaluator,
    validation_history,
    lr,
    elapsed,
    memory_usage,
    nfe,
    score_sign,
    val_metric_to_monitor,
):
    """Compile results into a single dict."""
    total_epochs = len(validation_history["loss.train"])

    # Setup a results dict
    results = OrderedDict(
        **{
            "lr": lr,
            "elapsed_time": elapsed,
            "memory_usage": memory_usage,
            "epochs_run": total_epochs,
            "time_per_epoch": elapsed / total_epochs,
            "num_params": get_num_params(model),
            "nfe": nfe,
            "nfe_per_epoch": nfe / total_epochs if nfe is not None else None,
        }
    )

    # Add best validation score
    func = np.argmax if score_sign == 1 else np.argmin
    best_idx = func(validation_history[val_metric_to_monitor + ".val"])
    for key, value in validation_history.items():
        results[key] = value[best_idx]

    # Add test results
    for metric, value in evaluator.state.metrics.items():
        results[metric + ".test"] = value

    return results


def add_metrics_to_dict(metrics, history, dot_str):
    """Adds metrics to a dict with the same keys.

    Args:
        metrics (dict): A dictionary of keys and lists, where the key is the metric name and the list is storing metrics
            over the epochs.
        history (dict): The history dict that contains the same keys as in metrics.
        dot_str (str): Metrics are labelled 'metric_name.{train/val}'.

    Returns:
        None
    """
    for name, value in metrics.items():
        history[name + dot_str].append(value)


def get_memory(device, reset=False, in_mb=True):
    """Gets the GPU usage."""
    if device is None:
        return float("nan")
    if device.type == "cuda":
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        bytes_ = torch.cuda.max_memory_allocated(device)
        if in_mb:
            bytes_ = bytes_ / 1024 / 1024
        return bytes_
    else:
        return float("nan")


def print_val_results(history, epoch=None, pbar=None, verbose=1):
    """Prints output of the validation loop to the console."""
    if verbose == 0:
        return None

    if epoch is None:
        print_string = "Final results:\n\t"
    else:
        print_string = "EPOCH: {}\n\t".format(epoch)

    # Get the keys we wish to print
    keys = list(np.unique([x.split(".")[0] for x in history.keys()]))

    for key in keys:
        keys_ = [key_ for key_ in history.keys() if key in key_]

        pstring = ""
        for i, k in enumerate(keys_):
            value = history[k] if epoch is None else history[k][-1]
            if "acc" in key:
                val_str = "{:.1f}%".format(value * 100)
            elif value is not None:
                val_str = "{:.3f}".format(value)
            else:
                val_str = None
            sep = " \t| " if i > 0 else ""
            pstring += sep + "{}: {}".format(k, val_str)
        print_string += pstring + "\n\t"

    if (verbose > 0) and (pbar is not None):
        pbar.write(print_string)


def save_results(_run, results, validation_history):
    """Save results dict as scalars and the validation history as a pickle file to the run dir."""
    for name, value in results.items():
        _run.log_scalar(name, value)


def get_num_params(model):
    """Gets the number of trainable parameters in a pytorch model."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
