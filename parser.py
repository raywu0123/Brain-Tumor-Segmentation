import argparse


def brain_tumor_argparse():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Experiment')

    general_args_group = parser.add_argument_group('General Arguments')
    add_general_args(general_args_group)

    optimizing_args_group = parser.add_argument_group('Optimizing Arguments')
    add_optimizing_args(optimizing_args_group)

    training_args_group = parser.add_argument_group('Training Arguments')
    add_training_args(training_args_group)

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
        '-async'
        '--async_load',
        dest='async_load',
        action='store_true',
        help='if True, use multiple processes to load data',
    )
    parser.add_argument(
        '--profile',
        dest='profile',
        action='store_true',
        help='if True, activate the profiler and dump the log'
    )
    parser.add_argument(
        '--preload',
        dest='preload',
        action='store_true',
        help='if True, preload the whole dataset before training'
    )
    parser.set_defaults(do_comet=False)
    parser.set_defaults(async_load=False)
    parser.set_defaults(profile=False)
    parser.set_defaults(preload=False)


def add_training_args(parser):
    parser.add_argument(
        '-lid',
        '--loss_function_id',
        type=str,
        default='crossentropy-log(dice)',
    )
    parser.add_argument(
        '-cg',
        '--clip_grad',
        type=float,
        default=0.,
        help='The gradient norm will be clipped by this param if it is greater than 0.'
    )
    parser.add_argument(
        '-obs',
        '--optim_batch_steps',
        type=int,
        default=1,
        help='Gradient accumulation for this many batches.',
    )
    parser.add_argument(
        '-aug',
        '--augmentation',
        dest='augmentation',
        action='store_true',
        help='if True, activate data augmentation while training',
    )
    parser.set_defaults(augmentation=False)


def add_optimizing_args(parser):
    parser.add_argument(
        '-lr',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '-ot',
        '--optimizer_type',
        type=str,
        default='Adam',
    )
    parser.add_argument(
        '-mil',
        '--epoch_milestones',
        type=int,
        nargs='+',
        default=[50, 70],
        help='learning rate scheduler milestone, unit: epoch'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.1,
        help='learning rate scheduler decay rate'
    )


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
    parser.set_defaults(save_volume=False)
