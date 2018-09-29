import argparse


def brain_tumor_argparse():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Experiment')
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
        '-g',
        '--use_generator',
        action='store_true',
        help='Use generator when RAM isn\'t large enough',
    )
    return parser
