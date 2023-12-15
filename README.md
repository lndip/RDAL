# Robust discriminative adversarial learning (RDAL)

The `RDAL` repository contains the code to reproduce the main results in the paper [Representation Learning for Audio Privacy Preservation Using Source Separation and Robust Adversarial Learning](https://ieeexplore.ieee.org/document/10248153/metrics#metrics) (WASPAA 2023)

## Structure
-  The folder `RDAL\data_final` contains the data to run the code.

- The folder `RDAL\pickle_data` contains the serialized data in pickle form after running `RDAL\src\serialize_data.py`.

- The folder `RDAL\src` contains the source code of RDAL+M.

- The folder `RDAL\models` contains the trained models in each run of each specific training mode. `RDAL\models\tsne_supervised` contains the trained models in the supervised training specifically for the TSNE visualization.

- The folder `RDAL\pickle_results` contains the results in pickle format of each run. `RDAL\best_pickle_results` contains the results of the best model in each approach (baseline, NaiveAdv, RDAL, and RDAL+M) for ROC curve plotting


## Requirement
The code is written in Python, and the models are implemented using Pytorch. Make sure all libraries in `environment.yml` are available.

## How to run
1. Check `README.md` file in `data_final` to obtain the data. Remove `.gitkeep` from the empty folders if needed.

2. Run

        python serialize_data.py
    to serialize the data. This file will calculate the short time Fourier trasform (STFT) spectrograms of both the merged signals and the sound event signals and then serialize the spectrograms with the sound event labels and speech labels into `.pickle` format for training the adversarial training process and the STFT spectrograms of the merged signals containing speech and their sound event correspondences for pre-training the source separtion network in RDAL+M setup.

3.  There are 4 setups provided in this package: The baseline, NaiveAdv, RDAL, and RDAL+M can be run from `baseline_main.py`, `naive_rdal_main.py`, `rdal_main.py`, and `rdal_mask_main.py` respectively. To run the model, run the following command:
        
        python {main_file_name}.py {job_idx}

4. The models from the training process of a specific training mode with a specific `job_idx` is saved in the folder `models`. The results are saved in the folder `pickle_results`, in `.pickle` format.

4. Result reading: 
- `read_pickle.ipynb` is provided to read the results from the pickle file. The plot of the predicted probability densities from the attacker model can be generated in the notebook.
- The TSNE plot can be generarted from `visualize_data.ipynb`. The trained model for the supervised training of the feature extractor on the sound event and the speech label can be found in the `models` directory.

## Reference
Please consider citing our paper if the work is useful for your research.
```
@INPROCEEDINGS{10248153,
author={Luong, Diep and Tran, Minh and Gharib, Shayan and Drossos, Konstantinos and Virtanen, Tuomas},
booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)}, 
title={Representation Learning for Audio Privacy Preservation Using Source Separation and Robust Adversarial Learning}, 
year={2023},
volume={},
number={},
pages={1-5}}
```

## Contact
- Diep Luong (lndiep1811@gmail.com)