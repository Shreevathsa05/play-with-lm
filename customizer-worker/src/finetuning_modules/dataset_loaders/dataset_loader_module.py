from datasets import load_dataset, Dataset
from typing import Optional, Union, Dict, Any

class ExternalDatasetLoader:
    """
    A utility class to load datasets from external sources (e.g., Hugging Face Hub, local files, URLs)
    using the 'datasets' library.
    """
    
    @staticmethod
    def load(
        path: str,
        name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Union[str, list, dict]] = None,
        **kwargs: Any
    ) -> Union[Dataset, Dict[str, Dataset]]:
        """
        Loads a dataset from the Hugging Face Hub, or a local dataset.
        
        Args:
            path (str): Path or name of the dataset. For Hugging Face datasets, this is the repo ID (e.g. 'imdb').
                        For local files, it can be the format ('csv', 'json', 'text') or directory path.
            name (str, optional): Defining the name of the dataset configuration.
            split (str, optional): Which split of the data to load. If None, will return a `DatasetDict` of all splits.
            data_files (str, list, dict, optional): Path(s) to source data file(s).
            **kwargs: Additional keyword arguments passed to `datasets.load_dataset`.
            
        Returns:
            Dataset or DatasetDict: The loaded dataset.
        """
        print(f"Loading dataset from: '{path}' (name={name}, split={split})...")
        
        try:
            dataset = load_dataset(
                path=path,
                name=name,
                split=split,
                data_files=data_files,
                **kwargs
            )
            print("Dataset successfully loaded!")
            return dataset
        except Exception as e:
            print(f"Error loading dataset '{path}': {e}")
            raise
