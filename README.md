# Brain-Tumor-Segmentation

## Usage

### 1. Install Dependencies
```
pipenv install
pipenv shell
```

### 2. Set up .env
``` 
cp .env.example .env
```

### 3. Run Exp. with Command Line

```
python main.py -m <model_id> -d <data_provider_id> [--comet]
```

* Please refer to `models/__init__.py` for available model_ids,
and `data/data_providers.py` for available data_provider_ids.  
* For other arguments, please refer to `parser.py`
* Passing the `--comet` argument allows the user to log results to comet.ml, 
you'll have to add your api-key to the `.env` file

### 4. Resume Training from Checkpoint

```
python main.py --checkpoint_dir <checkpoint_dir>
``` 

## 5. Prediction

```
python predict.py --checkpoint_dir <checkpoint_dir> [--predict_mode] [--save_volume]
```

## Data Configuration
### NTU

0. (Recommended) If you wish to preprocess with masks, run:
``` 
bash ./preprocess_tools/skull_strip.sh <bse_dir> <result_dir> <data_dir>
``` 
Where `bse_dir` is the binary file from the 
[BrainSuite](http://brainsuite.org/) package.


1. To preprocess, run:
```
cd preprocess_tools
python preprocess.py <data_dir> <mask_dir> <result_dir> 
```   
* This script will resize to $1mm^3$ resolution and center the brain with its mask.  
* The resulting shape is defined in the `ImageProcessor` class in `preprocess_tools/image_utils.py`

2. To postprocess, run:
``` 
python postprocess.py <lable_dir> <result_dir> <image_processor_path>
```
* This script will revert the resolution and copy the according headers from the original nii file.

### NTU2019
0. Directory Structure:
```
NTU2019
    image
    label
NTU2019_result
    image
    mask
```

### BRATS2015

0. Directory Structure:
``` 
BRATS2015
    BRATS2015_Training
    BRATS2015_Testing
```
* The inner structure is same as the original format downloaded from [https://www.smir.ch/BRATS/Start2015]().

### TCIA_CT

* Data from https://github.com/deepmind/tcia-ct-scan-dataset/tree/master/nrrds
* The corrensponding options for data_provider_id are
`tciact_oncologist`, `tciact_radiographer`, `tciact` for both

## Dependencies
* BrainSuite (optional)
* python 3.6
* pipenv
* python packages listed in `Pipfile` are managed by pipenv
