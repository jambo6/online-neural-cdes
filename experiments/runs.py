import logging
import os
from copy import deepcopy

import torch
from hyperoptimize import ex as ex_hyperopt
from main import ex as ex_main
from sacredex import run_over_configurations
from sacredex.parallel import JsonFolderIterator, dump_config_list_to_json_tmpfiles
from sacredex.run import check_if_run, purge_incomplete_runs
from utils import get_client, parse_configuration_json

EXPERIMENTS = {
    "hyperopt": ex_hyperopt,
    "interpolation": ex_main,
    "interpolation-v2": ex_main,
    "sparsity-v2": ex_main,
    "medical-sota-v2": ex_main,
}
CLIENT = get_client()


def run(run_name, gpus):
    """Run function for parallelising over GPUs.

    This function performs as follows:
        1. Builds the configuration according to the run name.
        2. Dumps the configurations into individual json files in a temporary directory.
        3. Runs a GNU parallel command that calls this files main __main__ script with the relevant CLIs to run the
        model over multiple GPUs.

    Args:
        run_name (str): The name of the run, this must be a key from the configuration json.
        gpus (list): List of GPUs to parallelise over.

    Returns:
        None
    """
    assert isinstance(gpus, list), "GPUs must be a list, this can be a 1-element list."
    assert max(gpus) < torch.cuda.device_count()

    # First build the configuration and get the test augmented run name
    configs, db_name = parse_configuration_json(run_name)

    # Add all configs to the client
    db = CLIENT[db_name]
    if "configs" in db.list_collection_names():
        db["configs"].drop()
    CLIENT[db_name]["configs"].insert_many(deepcopy(configs))
    purge_incomplete_runs(db)

    # Now delete the ones that are completed
    not_run_configs = []
    for c in configs:
        if not check_if_run(db, c):
            not_run_configs.append(c)
    if len(not_run_configs) == 0:
        raise Exception("All configurations have been run")

    # Save non-complete configs to a folder
    folder = dump_config_list_to_json_tmpfiles(not_run_configs)

    # Run gpu commands in parallel
    command = (
        "parallel -j {} -u --delay 5 --link --bar 'python ./runs.py {} {} {} -gpu {{1}}' ::: {}"
        "".format(
            len(gpus), run_name, db_name, folder, " ".join([str(x) for x in gpus])
        )
    )

    # Log the command and then run it
    logging.info("Running: {}".format(command))
    os.system(command)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="The name of the experiment to run.")
    parser.add_argument(
        "db_name", help="The name of the mongo db to store the info in.", default=-1
    )
    parser.add_argument(
        "folder", help="The folder containing the configuration jsons.", default=-1
    )
    parser.add_argument("-gpu", help="The index of the GPU to run on.", default=-1)
    args = parser.parse_args()

    # Setup ex
    assert args.run_name in EXPERIMENTS.keys(), "Unrecognised experiment"
    ex = EXPERIMENTS.get(args.run_name)

    # GPU is set via environment variables
    os.environ["GPU"] = str(args.gpu)

    # Configs
    configs = JsonFolderIterator(args.folder)

    # Run
    run_over_configurations(ex, configs, client=CLIENT, db_name=args.db_name)
