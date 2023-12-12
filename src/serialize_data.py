from typing import Union, MutableMapping
from pathlib import Path
from pickle import dump
import numpy as np
import librosa as lb
from utils import (get_files_from_dir_with_pathlib,
                  empty_dir,
                  NPY_DATA_SPLIT,
                  FREESOUND_DIR,
                  PICKLE_DATA,
                  FREESOUND_LABELS, 
                  SPEECH_LABELS,
                  GENDER_LABELS)


def stft_magnitude(audio_data: np.ndarray,
                    n_fft: int = 1411,
                    hop_length: int = 441) \
        -> np.ndarray:
    """Apply STFT on the audio data in time domain

    :param audio_file: Audio file in time domain.
    :type audio_file: np.ndarray
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 320.
    :type hop_length: Optional[int]
    :return: Augio signal in time- frequency domain
    :rtype: np.ndarray
    """
    stft = lb.stft(y=audio_data,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window='hamming')
    stft_mag = np.abs(stft)
    return stft_mag


def serialize_features_and_classes(pickle_path: Union[Path, str],
                                    features_and_classes: MutableMapping[str, Union[np.ndarray, int]]) \
        -> None:
    """ Serialize the features and classes

    :param pickle_path: Path of the pickle file
    :type pickle_path: Path|str
    :param features_and_classes: Features and classes.
    :type features_and_classes:dict[str, np.ndarray|int]  
    """
    with pickle_path.open('wb') as pickle_file:
        dump(features_and_classes, pickle_file)


def handle_one_data_pair(audio_path: Union[Path, str],
                        freesound_label: str,
                        speech_label: str,
                        sound_event_path: str,
                        output_pickle_path: Union[Path, str]) \
        -> None:
    """ Read data from time domain
        Create features_and_classes dict for the data and save it in a pickle file.

    :param audio_path: The path of the audio_data in time domain
    :type audio_path: Path|str
    :param freesound_label: The freesound label of the audio data
    :type freesound_label: str
    :param speech_label: The speech label of the data
    :type speech_label: str
    :param output_pickle_path: The pickle file path
    :type output_pickle_path: Path|str
    """
    features_and_classes = {}

    # Get the audio data in time domain
    audio_data = stft_magnitude(np.load(audio_path))
    sound_event_data = stft_magnitude(np.load(sound_event_path))

    # Get the label index for sound event and speech
    event = FREESOUND_LABELS.index(freesound_label)
    speech = SPEECH_LABELS.index(speech_label)
    gender = -1

    if(speech_label == "speech"):
        # Get the gender label
        if '_female_' in audio_path.stem:
            gender = GENDER_LABELS.index('female')
        elif '_male_' in audio_path.stem:
            gender = GENDER_LABELS.index('male')

    # Assign value to the dict
    features_and_classes["features"] = audio_data 
    features_and_classes["freesound_class"] = event
    features_and_classes["speech_class"] = speech
    features_and_classes["sound_event_data"] = sound_event_data
    features_and_classes["gender_class"] = gender

    serialize_features_and_classes(Path(output_pickle_path), features_and_classes)


def create_pickle_data(split: str) -> None:
    """Create pickle data for the chosen split set

    :param split: The dataset split
    :type split: str
    """
    # Get the split's directory path
    numpy_dir = Path.joinpath(NPY_DATA_SPLIT, split)
    pickle_dir = Path.joinpath(PICKLE_DATA, split)

    empty_dir(pickle_dir)

    for freesound_label in FREESOUND_LABELS:
        for speech_label in SPEECH_LABELS:
            freesound_speech_dir = Path.joinpath(numpy_dir, freesound_label, speech_label)
            audio_files = get_files_from_dir_with_pathlib(freesound_speech_dir)
            for audio_path in audio_files:
                sound_event_file = audio_path.stem.split('_')[0] +'_'+ audio_path.stem.split('_')[1] +'_'+ audio_path.stem.split('_')[2] + '.npy'
                sound_event_path = Path(FREESOUND_DIR, split, freesound_label, sound_event_file) if split=="eval" else Path(FREESOUND_DIR, "dev", freesound_label, sound_event_file)
                output_pickle_path = Path.joinpath(pickle_dir, f'{audio_path.stem}.pickle')
                handle_one_data_pair(audio_path, freesound_label, speech_label, sound_event_path, output_pickle_path)


def main():
    for split in ["train", "val", "eval"]:
        create_pickle_data(split)


if __name__ == "__main__":
    main()


