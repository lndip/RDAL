# Robust discriminative adversarial learning with masking (source code)

The source code for Robust Discrminative Adversarial Learning with Masking (RDAL+M).

## Requirement
The code is written in Python, and the models are implemented using Pytorch. Make sure all libraries in `environment.yml` are available.

## How to run
1. Run

        python serialize_data.py
    to serialize the data. This file will calculate the short time Fourier trasform (STFT) spectrograms of both the merged signals and the sound event signals and then serialize the spectrograms with the sound event labels and speech labels into `.pickle` format for training the adversarial training process and the STFT spectrograms of the merged signals containing speech and their sound event correspondences for pre-training the source separtion network. The pickle data is also provided in the dataset.
2.  There are 4 setups provided in this package: The baseline, NaiveAdv, RDAL, and RDAL+M can be run from `baseline_main.py`, `naive_rdal_main.py`, `rdal_main.py`, and `rdal_mask_main.py` respectively. To run the model, run the following command:
        
        python {main_file_name}.py {job_idx}

3. The models from the training process of a specific training mode with a specific `job_idx` is saved in the folder `models`. The results are saved in the folder `pickle_results`, in `.pickle` format.

4. Result reading: 
- `read_pickle.ipynb` is provided to read the results from the pickle file. The plot of the predicted probability densities from the attacker model can be generated in the notebook.
- The TSNE plot can be generarted from `visualize_data.ipynb`. The trained model for the supervised training of the feature extractor on the sound event and the speech label can be found in the `models` directory.

## Authors
- Diep Luong (diep.luong@tuni.fi)
- Minh Tran
- Shayan Gharib