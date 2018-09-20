# Brain-Tumor-Segmentation
Experiment of brain tumor segmentation

## Usage

### 1. Install Dependencies

If you want to run it with gpu,

```
pipenv install
pipenv shell
```


### 2. Set up .env

``` 
cp .env.sample .env
```

### 3. Run Exp. with Command Line

```
python main.py -m <model_id> -d <data_provider_id>
```

### 4. Syntax Lynting

```
python -m flake8 .
```

## Dependencies
* pipenv
* python packages are managed by pipenv