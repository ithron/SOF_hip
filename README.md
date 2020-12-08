# Tensorflow Dataset (TFDS) Wrapper for the SOF Dataset

A TFDS wrapper for the SOF dataset.
Information on how to access the SOF dataset can be found on the [SOF Online website](https://sleepdata.org/datasets/sof).

## Dependencies
Requires `tensorflow_datasets >= 4.1.0`, `tqdm` and `sof_utils>=0.0.5`.
```shell
pip install tfds-nightly tqdm sof_utils>=0.0.5
```


## Building
Automatic downloading is not supported since the data is not publicly available.

From inside the `SOF_hip` directory:
```shell
tfds build --manual_dir /path/to/SOF/Dataset
```

## Configurations
- `unsupervised_raw` unprocessed radiographs without any labels
- `unsupervised_raw_tiny` same as `unsupervised_raw` but with only 1000 examples

## Usage
```python
import tensorflow_dataset as tfds
import SOF_hip

tfds.load('SOF_hip/unsupervised_raw')
```
## License

This wrapper is licensed under BSD 3-Clause License.
__This license does not apply for the SOF dataset, only for the wrapper in this repository__ 