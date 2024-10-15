#!/usr/bin/env python
import os  # noqa: E402
os.environ["COMET_URL_OVERRIDE"] = "https://www.comet-ml.com/clientlib/"
os.environ["COMET_WS_URL_OVERRIDE"] = "wss://www.comet-ml.com/ws/logger-ws"

import comet_ml  # noqa: F401 E402
from comet_ml import Optimizer  # noqa: E402

import torch  # noqa: E402


from TrackToLearn.trainers.cosine_distance_train import (  # noqa: E402
    parse_args,
    CosineDistanceTrackToLearnTraining)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert(torch.cuda.is_available())


def main():
    """ Main tracking script """
    args = parse_args()
    print(args)

    # We only need to specify the algorithm and hyperparameters to use:
    config = {
        # We pick the Bayes algorithm:
        "algorithm": "grid",

        # Declare your hyperparameters in the Vizier-inspired format:
        "parameters": {
            "lr": {
                "type": "discrete",
                "values": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]},
            "n_dirs": {
                "type": "discrete",
                "values": [2, 4, 10]},
            "dropout": {
                "type": "discrete",
                "values": [0.0, 0.25, 0.5, 0.75]},
            "hidden_dims": {
                "type": "categorical",
                "values": ["512-512-512", "1024-1024"]}

        },

        # Declare what we will be optimizing, and how:
        "spec": {
            "metric": "Reward",
            "objective": "maximize",
            "seed": args.rng_seed,
        },
    }

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config, verbose=True)
    for experiment in opt.get_experiments(
        project_name=args.experiment, auto_metric_logging=False,
        workspace='TrackToLearn', parse_args=False
    ):

        lr = experiment.get_parameter("lr")
        dropout = experiment.get_parameter("dropout")
        hidden_dims = experiment.get_parameter("hidden_dims")
        n_dirs = experiment.get_parameter("n_dirs")

        cosine_distance_experiment = CosineDistanceTrackToLearnTraining(
            # Dataset params
            args.path,
            args.experiment,
            args.name,
            args.dataset_file,
            args.reference_file,
            args.scoring_data,
            # RL params
            args.max_ep,
            args.gradient_steps_per_epoch,
            lr,
            dropout,
            args.batch_size,
            # Env params
            n_dirs,
            args.interface_seeding,
            # Tracking params
            args.npv,
            args.theta,
            args.min_length,
            args.max_length,
            args.step_size,
            # Model params
            hidden_dims,
            args.add_neighborhood,
            # Experiment params
            args.precomputed_states,
            args.num_workers,
            args.use_gpu,
            args.rng_seed,
            experiment,
            args.run_tractometer,
            args.render
        )
        try:
            cosine_distance_experiment.run()
            experiment.end()
        except comet_ml.exceptions.InterruptedExperiment as e:
            print(e)


if __name__ == '__main__':
    main()
