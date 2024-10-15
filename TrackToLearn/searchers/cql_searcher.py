#!/usr/bin/env python
import os  # noqa: E402
os.environ["COMET_URL_OVERRIDE"] = "https://www.comet-ml.com/clientlib/"
os.environ["COMET_WS_URL_OVERRIDE"] = "wss://www.comet-ml.com/ws/logger-ws"

import comet_ml  # noqa: F401 E402
from comet_ml import Optimizer  # noqa: E402

import torch  # noqa: E402


from TrackToLearn.trainers.cql_train import (  # noqa: E402
    parse_args,
    CQLTrackToLearnTraining)
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
                "values": [6e-5, 9e-5, 1e-4, 3e-4, 6e-4]},
            "gamma": {
                "type": "discrete",
                "values": [0.99]},
            "alpha": {
                "type": "discrete",
                "values": [0.5, 1.0, 2.0]},
            "min_q_weight": {
                "type": "discrete",
                "values": [1.0, 5.0, 10.0]},
            "backup_entropy": {
                "type": "discrete",
                "values": [True, False]},
            "automatic_entropy_tuning": {
                "type": "discrete",
                "values": [True, False]},
            "n_dirs": {
                "type": "discrete",
                "values": [4]},
            "hidden_dims": {
                "type": "categorical",
                "values": ["1024-1024"]}

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
        gamma = experiment.get_parameter("gamma")
        alpha = experiment.get_parameter("alpha")
        min_q_weight = experiment.get_parameter("min_q_weight")
        backup_entropy = experiment.get_parameter("backup_entropy")
        automatic_entropy_tuning = experiment.get_parameter(
            "automatic_entropy_tuning")
        hidden_dims = experiment.get_parameter("hidden_dims")
        n_dirs = experiment.get_parameter("n_dirs")

        cql_experiment = CQLTrackToLearnTraining(
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
            gamma,
            alpha,
            min_q_weight,
            backup_entropy,
            automatic_entropy_tuning,
            args.batch_size,
            # Env params
            n_dirs,
            args.dense_rewards,
            args.reward_scaling,
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
            cql_experiment.run()
            experiment.end()
        except comet_ml.exceptions.InterruptedExperiment as e:
            print(e)


if __name__ == '__main__':
    main()
