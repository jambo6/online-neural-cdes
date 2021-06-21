# flake8: noqa
import logging

import sacred
from ax.service.managed_loop import optimize
from ax.service.utils.best_point import get_best_raw_objective_point
from ingredients.loader import data_ingredient, load_data
from ingredients.setup_model import model_ingredient, setup_model
from ingredients.trainer import train, train_ingredient
from sacredex import run_over_configurations
from utils import get_client, parse_configuration_json

# Remove ax info logging
logging.disable(logging.INFO)

# Initialise sacred experiment
sacred.SETTINGS["CAPTURE_MODE"] = "sys"
ex = sacred.Experiment(
    "hyperopt", ingredients=[data_ingredient, model_ingredient, train_ingredient]
)


@ex.config
def config():
    seed = 0
    test_mode = False
    dataset_name = None
    hyperparameters = None
    evaluation_metric = None
    minimize = None
    total_trials = None
    parameterization = None


@ex.main
def main(
    _run, seed, test_mode, evaluation_metric, minimize, total_trials, parameterization
):
    # Load
    (
        (train_dl, val_dl, test_dl),
        input_dim,
        output_dim,
        static_dim,
        model_interpolation,
        return_sequences,
    ) = load_data(test_mode=test_mode)

    def train_evaluate(parameterization):
        # Open hyperparameters and save to _run.info
        model_params, trainer_params = handle_parameterization(parameterization, _run)

        # Get the model
        model, prepare_batch = setup_model(
            input_dim,
            output_dim,
            static_dim,
            model_interpolation,
            **model_params,
            return_sequences=return_sequences,
            train_dl=train_dl
        )

        # Train
        try:
            model, results, _ = train(
                _run,
                model,
                train_dl,
                val_dl,
                test_dl,
                prepare_batch=prepare_batch,
                verbose=1,
            )
            metric_value = results["{}.val".format(evaluation_metric)]
        except:
            # This is hacky but ax cant deal with nans
            metric_value = 1000 if minimize else 0

        # Careful! ax requires specification of the value *and* the stddev in the format below
        ax_output = {evaluation_metric: (metric_value, 0.0)}

        return ax_output

    # Optimize, open from experiment as something odd goes on under the hood with ax when extracting the value
    _, v, experiment, _ = optimize(
        parameters=parameterization,
        evaluation_function=train_evaluate,
        experiment_name="hyperopt",
        objective_name=evaluation_metric,
        total_trials=total_trials,
        minimize=minimize,
        random_seed=seed,
    )

    # Save best params
    best_params, metric_info = get_best_raw_objective_point(experiment)
    metric = metric_info[evaluation_metric][0]
    _run.info["best_parameters"] = undo_dunder(best_params)
    _run.info[evaluation_metric] = metric


def handle_parameterization(parameterization, _run=None):
    # Opens the parametrization into the model and trainer params
    assert [
        "__" in x for x in parameterization.keys()
    ], "All parameters must be dundered."
    model_params, trainer_params = {}, {}
    for key, value in parameterization.items():
        outer_key, inner_key = key.split("__")
        if outer_key == "model":
            model_params[inner_key] = value
        elif outer_key == "trainer":
            trainer_params[inner_key] = value
        else:
            raise NotImplementedError(
                "Only implemented for model and trainer ingredients, got {}".format(
                    outer_key
                )
            )
    # Save params
    if _run is not None:
        for d in (model_params, trainer_params):
            for key, value in d.items():
                _run.log_scalar(key, value)
    return model_params, trainer_params


def undo_dunder(parameters):
    # Convert ax output back to sacred ingredient style nested dict
    output_dict = {
        key: {} for key in set([x.split("__")[0] for x in parameters.keys()])
    }
    for k, v in parameters.items():
        outer_key, inner_key = k.split("__")
        output_dict[outer_key][inner_key] = v
    return output_dict


if __name__ == "__main__":
    from sacredex import run_over_configurations
    from utils import get_client, parse_configuration_json

    # client = get_client()
    client = None

    # Configuration needs some logic so that we dont get repeated runs
    config_list, run_name = parse_configuration_json("hyperopt")
    config_list = [c for c in config_list if config_list["dataset_name"] == 'CharacterTrajectories']

    # Run
    run_over_configurations(
        ex,
        config_list,
        client=client,
        db_name="delete",
    )
