# flake8: noqa
from ingredients.loader import data_ingredient, load_data
from ingredients.setup_model import model_ingredient, setup_model
from ingredients.trainer import train, train_ingredient
from sacred import Experiment

# Initialise sacred experiment
ex = Experiment(
    "main", ingredients=[data_ingredient, model_ingredient, train_ingredient]
)


@ex.config
def config():
    seed = 0
    test_mode = False
    dataset_name = None
    hyperparameters = None


@ex.main
def main(_run, test_mode):
    # Load
    (
        (train_dl, val_dl, test_dl),
        input_dim,
        output_dim,
        static_dim,
        model_interpolation,
        return_sequences,
    ) = load_data(test_mode=test_mode)

    # Get the model
    model, prepare_batch = setup_model(
        input_dim,
        output_dim,
        static_dim,
        model_interpolation,
        return_sequences=return_sequences,
        train_dl=train_dl,
    )

    # Train
    model, results, _ = train(
        _run, model, train_dl, val_dl, test_dl, prepare_batch=prepare_batch, verbose=1
    )


if __name__ == "__main__":
    from sacredex import run_over_configurations
    from utils import get_client, parse_configuration_json

    # Configuration needs some logic so that we dont get repeated runs
    config_list, run_name = parse_configuration_json("interpolation")

    # Run
    client = get_client()
    client = None
    run_over_configurations(
        ex,
        config_list,
        client=client,
        db_name=run_name,
    )
