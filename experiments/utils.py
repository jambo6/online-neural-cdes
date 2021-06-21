import logging

import json5
import variables
from pymongo import MongoClient
from sacredex.utils import nested_parameter_grid


def load_json(file):
    """Load a json file into a dict."""
    with open(file) as json_file:
        data = json5.load(json_file)
    return data


def get_client():
    return MongoClient(host=variables.ATLAS_HOST)


def load_datasets_json():
    return load_json("./configurations/dataset.json5")


def parse_dataset_json(dataset_name):
    """Get specialised info from the dataset.json"""
    # Load the main cfg
    cfg = load_datasets_json().get(dataset_name)

    # To store updated config values
    new_config = dict.fromkeys(["dataset", "trainer", "model"])

    # Get the data bits
    new_config["dataset"] = {
        x: cfg[x]
        for x in ["data_loc", "problem", "use_static", "evaluation_metric", "minimize"]
    }

    # Model cfg is just return sequences if online
    new_config["model"] = {
        "return_sequences": True if cfg["problem"] == "online" else False
    }

    # Get the train bits
    new_config["trainer"] = {x: cfg[x] for x in ["loss_str", "metrics"]}
    new_config["trainer"]["val_metric_to_monitor"] = cfg["evaluation_metric"]

    return new_config


def update_hyperparams(parameters, run_name="hyperopt"):
    """Update the hyperopt results for every config in a config list."""
    # We load all hyperopt runs, there are probably better ways to do this, but it is much faster than querying once
    # per parameter set.
    db = get_client()[run_name]
    hyperopt_runs = list(db.runs.find())

    # So far we only need to match the interpolation scheme
    for p in parameters:
        if p["hyperparameters"] in ["hyperopt", "set"]:
            continue

        hyperopt = []
        for h in hyperopt_runs:
            # Smoothing schemes to use linear hyperparams
            interpolation = p["dataset"]["interpolation"]
            if interpolation in (
                "linear_cubic_smoothing",
                "linear_quintic_smoothing",
                "linear_forward_fill",
                "cubic_forward_fill",
            ):
                interpolation = "linear"

            if all(
                [
                    h["config"]["dataset"]["interpolation"] == interpolation,
                    h["config"]["dataset_name"] == p["dataset_name"],
                    h["config"]["model"]["model_string"] == p["model"]["model_string"],
                    h["config"]["model"]["adjoint"] == p["model"]["adjoint"],
                ]
            ):
                hyperopt.append(h)

        if len(hyperopt) == 0:
            raise FileNotFoundError("Cannot find hyperopt for config\n{}".format(p))
        elif len(hyperopt) > 1:
            raise KeyError(
                "Multiple files found for config\n{}\nRun ids are: {}".format(
                    p, [h["_id"] for h in hyperopt]
                )
            )

        best_parameters = hyperopt[0]["info"]["best_parameters"]

        # Just in case
        run_id = hyperopt[0]["_id"]
        assert (
            best_parameters is not None
        ), "Cannot find best parameters information for hyperopt run {}".format(run_id)

        # Update p
        for key, value in best_parameters.items():
            if isinstance(value, dict):
                p[key].update(value)
            else:
                p[key] = value

    return parameters


def _update_nested_config(config, updater_dict):
    """Updates a config of the standard nested format."""
    for key, value in updater_dict.items():
        if isinstance(value, dict):
            if key in config:
                config[key].update(**value)
            else:
                config[key] = value
        else:
            config[key] = value
    return config


def set_test_mode(config, run_name=None, hyperopt_run_name=None):
    """Set config for testing."""
    config["test_mode"] = [True]
    if "total_trials" in config:
        config["total_trials"] = [2]
    if "trainer" in config:
        config["trainer"]["max_epochs"] = [10]
    else:
        config["trainer"] = {"max_epochs": [10]}

    logging.log(logging.WARNING, "Running in test mode!")

    # Prepend test to run name
    if run_name is not None:
        run_name = "test_{}".format(run_name)
    if hyperopt_run_name is not None:
        hyperopt_run_name = "test_{}".format(hyperopt_run_name)

    return config, run_name, hyperopt_run_name


def parse_configuration_json(run_name, hyperopt_run_name="hyperopt"):
    """Parses the configuration to a dictionary, adds dataset specifics and hyperparameters if specified.

    Args:
        run_name (str): The name of the experiment.
        hyperopt_run_name (str): The database containing the hyperopt runs.

    Returns:
        A list of config_list, these have already gone through sklearn grid style processing.
    """
    cfg = load_json("./configurations/configurations.json5")
    assert run_name in cfg.keys(), "Configuration not found for {}".format(run_name)
    cfg = cfg[run_name]

    # Make test mode if test
    db_name = run_name
    if variables.TEST_MODE:
        cfg, db_name, hyperopt_run_name = set_test_mode(
            cfg, run_name, hyperopt_run_name
        )

    # Hyperparameter method must be given, either hyperopt run, load them, or they are already set.
    assert cfg["hyperparameters"] in [["hyperopt"], ["load"], ["set"]]

    # If a multi-config is specified, expand out into appropriate lists
    multi_configs = cfg.get("multi-config")
    if multi_configs:
        del cfg["multi-config"]

        # Create the grid
        params = []
        for key in multi_configs.keys():
            run_dict = multi_configs[key]
            run_dict.update(cfg)
            params.extend(nested_parameter_grid(run_dict))
    else:
        params = nested_parameter_grid(cfg)

    # Now we need to add the dataset specific options and the hyperopt options to the dataset
    updated_params = []
    for p in params:
        # Update with data specific options
        dataset_config = parse_dataset_json(p["dataset_name"])
        p = _update_nested_config(p, updater_dict=dataset_config)

        # If the run name is hyperopt, add some info to ensure the experiment knows its a hyperopt run
        if cfg["hyperparameters"] == ["hyperopt"]:
            # Make sure evaluation metric is top level so hyperopt can access it
            p["evaluation_metric"] = p["dataset"]["evaluation_metric"]
            p["minimize"] = p["dataset"]["minimize"]

        updated_params.append(p)

    # Now to update the hyperparameters if they are set
    # This is done out of the above loop so it does not require len(params) calls to Atlas
    updated_params = update_hyperparams(updated_params, run_name=hyperopt_run_name)

    return updated_params, db_name


# client = get_client()
# from sacredex.run import _delete_run_id
#
# db = client["sparsity-v2"]
# a = db.runs.find(
#     {
#         "config.dataset_name": {
#             "$in": [
#                 "BeijingPM3pt5",
#                 "BejiingPM10",
#                 "CharacterTrajectories",
#                 "SpeechCommands",
#             ]
#         }
#     }
# )
# for x in a:
#     print(x["_id"])
#     _delete_run_id(db, x["_id"])
