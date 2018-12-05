import os
import numpy as np
from dotenv import load_dotenv
# from tqdm import tqdm
# import nibabel as nib

from parser import brain_tumor_argparse
parser = brain_tumor_argparse()
args = parser.parse_args()

from models import ModelHub
from data.data_generator_factories import DataProviderHub
from utils import parse_exp_id
load_dotenv('./.env')


def flow(
        data_provider_id,
        data_provider,
        model,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    test_volumes = data_provider.get_testing_data()
    predict_probs = model.predict(
        test_volumes,
        **{
            **fit_hyper_parameters,
            'verbose': True,
        }
    )

    label = test_volumes['label']
    print(predict_probs.shape, label.shape)
    print(np.max(predict_probs), np.min(predict_probs), np.mean(predict_probs))
    print(np.max(label), np.min(label), np.mean(label))

    data_provider.metric(predict_probs, label).all_metrics()
    # predict_binarys = (predict_probs > 0.5).astype(float)
    #
    # predict_binarys = predict_binarys[:, 0, :, :]
    # predict_probs = predict_probs[:, 0, :, :]
    # # (image_id, D, H, W) to (image_id, H, W, D)
    # predict_probs = np.transpose(predict_probs, (0, 2, 3, 1))
    # predict_binarys = np.transpose(predict_binarys, (0, 2, 3, 1))
    #
    # test_ids = data_provider.all_ids
    #
    # binary_prediction_path = os.path.join(
    #     args.checkpoint_dir, f'binary_predict_on_{data_provider_id}'
    # )
    # if not os.path.exists(binary_prediction_path):
    #     os.mkdir(binary_prediction_path)
    #
    # prob_prediction_path = \
    # os.path.join(args.checkpoint_dir, f'prob_predict_on_{data_provider_id}')
    # if not os.path.exists(prob_prediction_path):
    #     os.mkdir(prob_prediction_path)
    #
    # for predict_prob, predict_binary, test_id in tqdm(
    #         zip(predict_probs, predict_binarys, test_ids)
    # ):
    #     stripped_test_id = test_id.strip('.npy').strip('.nii.gz')
    #     binary_save_path = os.path.join(binary_prediction_path, f'{stripped_test_id}.nii.gz')
    #     data_provider.save_result(predict_binary, binary_save_path, test_id)
    #     prob_save_path = os.path.join(prob_prediction_path, f'{stripped_test_id}.nii.gz')
    #     data_provider.save_result(predict_prob, prob_save_path, test_id)


if __name__ == '__main__':
    exp_id = os.path.basename(os.path.normpath(args.checkpoint_dir))
    model_id, data_id, time_stamp = parse_exp_id(exp_id)
    os.environ['EXP_ID'] = exp_id

    get_data_provider, data_provider_parameters = DataProviderHub[args.data_provider_id]
    data_provider = get_data_provider(data_provider_parameters)

    get_model, fit_hyper_parameters = ModelHub[model_id]
    model = get_model(
        **data_provider.get_data_format(),
    )
    model.load(args.checkpoint_dir)

    flow(
        data_provider_id=args.data_provider_id,
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
    )
