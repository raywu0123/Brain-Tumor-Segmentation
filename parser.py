import argparse


def brain_tumor_argparse():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Experiment')

    general_args_group = parser.add_argument_group('General Arguments')
    add_general_args(general_args_group)

    prediction_args_group = parser.add_argument_group('Prediction Arguments')
    add_prediction_args(prediction_args_group)
    return parser


def add_general_args(parser):
    parser.add_argument(
        '-m',
        '--model_id',
        type=str,
        help='The model_id to be used.',
    )
    parser.add_argument(
        '-d',
        '--data_provider_id',
        type=str,
        help='The medical-image data provider.',
    )
    parser.add_argument(
        '-grs',
        '--global_random_seed',
        type=int,
        help='The global random seed. [5566]',
        default=5566,
    )
    parser.add_argument(
        '--comet',
        dest='do_comet',
        action='store_true',
        help='Use comet-ml to document the info.',
    )
    parser.add_argument(
        '-cop',
        '--comet_project',
        type=str,
        help='comet ml project name',
        default='braintumorbaba',
    )
    parser.add_argument(
        '-cow',
        '--comet_workspace',
        type=str,
        help='comet ml workspace name',
        default='raywu0123',
    )
    parser.add_argument(
        '-aug',
        '--augmentation',
        type=bool,
        help='if True, activate data augmentation while training',
    )
    parser.add_argument(
        '-async'
        '--async_load',
        type=bool,
        help='if True, use multiple processes to load data'
    )
    parser.set_defaults(augmentation=False)
    parser.set_defaults(async_load=False)
    parser.set_defaults(save_volume=False)


def add_prediction_args(parser):
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='checkpoint directory to load the model',
    )
    parser.add_argument(
        '--predict_mode',
        type=str,
        help='Choose to predict on full dataset or only 1/10, [full/test]',
        default='test',
    )
    parser.add_argument(
        '--save_volume',
        dest='save_volume',
        action='store_true',
        help='Runs faster if not saving volumes.',
    )
    parser.set_defaults(do_comet=False)
