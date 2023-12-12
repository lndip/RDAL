from pathlib import Path
from typing import Optional, Union
from dataset_class import FreesoundSpeechDataset
from torch.utils.data import DataLoader, Dataset

PICKLE_DATA = Path("rdal-m_temp", "pickle_data")


def get_dataset(data_split: Union[str, Path],
                data_dir: Optional[str] = '',
                load_into_memory: Optional[bool] = True) \
        -> None:
    """Create and return a dataset, according to FreesoundSpeechDataset

    :param data_split: The data split
    :type data_split: str|Path
    :param data_dir: Directory to read the data from
    :type data_dir: str|Path
    :param load_into_memory: Load the data into memory?
    :type load_into_memory: bool
    :return: Dataset
    :rtype: dataset_class.FreesoundSpeechDataset
    """
    return FreesoundSpeechDataset(data_split=data_split,
                                  data_dir=data_dir,
                                  load_into_memory=load_into_memory)

        
def get_data_loader(dataset: Dataset,
                    batch_size: int,
                    shuffle: bool,
                    drop_last: bool) \
        -> DataLoader:
    """Create and return a data loader

    :param dataset: Dataset to use
    :type dataset: torch.ulis.data.Dataset
    :param batch_size: Batch size to use
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :param drop_last: Drop last?
    :type drop_last: bool
    :return: Data loader, using the specified dataset
    :rtype: torch.ulis.data.Dataloader
    """
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=1)


def get_all_data_loaders(batch_size: int)-> DataLoader:
    """Get all dataloaders for the training, validation, and testing

    :param batch_size: Batch size
    :type batch_size: int
    :return: The training dataloader, validation dataloader, and the test dataloader
    :rtype: torch.ulis.data.Dataloader
    """
    train_data_loader = get_data_loader(get_dataset('train', PICKLE_DATA), batch_size=batch_size, shuffle=True, drop_last=False)
    val_data_loader = get_data_loader(get_dataset('val', PICKLE_DATA), batch_size=batch_size, shuffle=False, drop_last=False)
    test_data_loader = get_data_loader(get_dataset('eval', PICKLE_DATA), batch_size=batch_size, shuffle=False, drop_last=False)

    return train_data_loader, val_data_loader, test_data_loader
