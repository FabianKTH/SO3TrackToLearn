#!/usr/bin/env python
import comet_ml  # noqa: F401 ugh
from comet_ml import Optimizer
import torch

from TrackToLearn.trainers.td3_train import (
    parse_args,
    TD3TrackToLearnTraining)

def main(config):
    """ Main tracking script """
    args = parse_args()

    # Next, create an optimizer, passing in the config:
    opt = Optimizer(config, project_name=args.experiment)

    for experiment in opt.get_experiments():
        experiment.auto_metric_logging = False
        experiment.workspace = 'fabiankth_ttl'
        experiment.parse_args = False
        experiment.disabled = not args.use_comet

        arguments = vars(args)

        # update optimization parameters
        arguments.update(
            {param: experiment.get_parameter(param)
             for param in config['parameters'].keys()}
            )

        print(arguments)

        td3_experiment = TD3TrackToLearnTraining(
            arguments,
            experiment
        )
        td3_experiment.run()
