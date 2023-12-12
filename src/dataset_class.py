from typing import Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

from utils import get_files_from_dir_with_pathlib

class FreesoundSpeechDataset(Dataset):
    def __init__(self,
                data_split: Union[Path, str],
                data_dir: Optional[Union[Path,str]]="",
                load_into_memory: Optional[int]=True) \
             -> None:
        """ The dataset object for the freesound and speech data

        :param data_split: The data split.
        :type data_split: str
        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        data_path = Path(data_dir, data_split)
        self.files = get_files_from_dir_with_pathlib(data_path)

        # Categorize the files into speech/no_speech
        self.speech_files = []
        self.no_speech_files = []

        for file in self.files:
            if "no_speech" in file.stem:
                self.no_speech_files.append(file)
            else:
                self.speech_files.append(file)

        self.load_into_memory = load_into_memory
        self.key_features = "features"
        self.key_freesound_class = "freesound_class"
        self.key_speech_class = "speech_class"
        self.sound_event_data = "sound_event_data"
        self.key_gender_class = "gender_class"
        if self.load_into_memory:
            for i, file in enumerate(self.no_speech_files):
                self.no_speech_files[i] = self._load_file(file)

            for i, file in enumerate(self.speech_files):
                self.speech_files[i] = self._load_file(file)


    @staticmethod
    def _load_file(file_path: Path)\
            -> Dict[str, Union[int, np.ndarray]]:
        """ Load the dictionary containing the data from a pickle file

        :param file_path: File path from which to load the dict
        :type file_path: Path
        :return: The dict containing the data
        :rtype: dict(str, int|np.ndarray)
        """
        with file_path.open('rb') as file:
            return pickle_load(file)


    def __len__(self) -> int:
        """Return the length of the dataset

        :return: Length of the dataset
        :rtype: int
        """
        return np.min([len(self.speech_files), len(self.no_speech_files)])

    
    def __getitem__(self, index:int) \
            -> tuple:
        """Return an item from the dataset

        :param index: Index of the item
        :type index: int
        :return: Features and classes of the item
        :rtype: tuple
        """
        if self.load_into_memory:
            the_item_speech = self.speech_files[index]
            the_item_no_speech = self.no_speech_files[index]
        else:
            the_item_speech = self._load_file(self.speech_files[index])
            the_item_no_speech = self._load_file(self.no_speech_files[index])
        return the_item_speech[self.key_features], \
               the_item_speech[self.key_freesound_class], \
               the_item_speech[self.key_speech_class], \
               the_item_speech[self.sound_event_data], \
               the_item_speech[self.key_gender_class],\
                the_item_no_speech[self.key_features], \
                the_item_no_speech[self.key_freesound_class], \
                the_item_no_speech[self.key_speech_class], \
                the_item_no_speech[self.sound_event_data], \
                the_item_no_speech[self.key_gender_class]


