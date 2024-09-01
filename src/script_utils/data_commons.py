
import os
from typing import Optional, List, Type, Union, Any
import hashlib
import pickle

from data.structure.chronology import Chronology
from data.structure.loaders import Loader
from data.dataset import SeriesDataset
from data.constants import DATA_STRUCTURES_SAVE_FOLDER_DEFAULT


class DataManager():

    @staticmethod
    def folder_to_files(folder : str) -> List[str]:
        if os.path.isdir(folder):
            return [os.path.join(folder, name) for name in os.listdir(folder)]
        else:
            return [folder]
        

    def __init__(self, save_folder : str = DATA_STRUCTURES_SAVE_FOLDER_DEFAULT, saving_allowed : bool = True, force_data_computation : bool = False, path_components_in_save_name : int = 2):
        self.save_folder = save_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        self.saving_allowed = saving_allowed
        self.force_data_computation = force_data_computation
        self.path_components_in_save_name = path_components_in_save_name

        self.save_types = {
            "chronology" : {
                "class" : Chronology,
                "loading_func": Chronology.deserialize,
                "extension" : "json",
            },
            "series_dataset" : {
                "class" : SeriesDataset,
                "loading_func": SeriesDataset.load,
                "extension" : "pt",
            },
            "pickle" : {
                "class" : dict,
                "loading_func": pickle.load,
                "extension" : "pickle",
            }
        }


    def get_savetype(self, savetype : Type) -> str:
        for key, value in self.save_types.items():
            if value["class"] == savetype:
                return key
        raise ValueError(f"Save type {savetype} not found in {list(self.save_types.keys())}.")

    def get_savepath(self, path : str, savetype : str, loadertype : Optional[str] = None) -> str:
        path_hash = hashlib.md5(path.encode()).hexdigest() # Hash is used to avoid conflicts in save names
        path_components = os.path.normpath(path).split(os.sep)
        lastname = "_".join(path_components[-self.path_components_in_save_name:]) # Last path components are used in the save name. Default is 2 to match e.g. "DATASET_NAME/DATASET_SPLIT"

        filename = f"{lastname}_{savetype}{'_' + loadertype if loadertype is not None else ''}_{path_hash}.{self.save_types[savetype]['extension']}"
        return os.path.join(self.save_folder, filename)
    

    def create_chronology(self, path : str, chronology_kwargs : dict) -> Chronology:
        files = DataManager.folder_to_files(path)
        chronology = Chronology.create(files, **chronology_kwargs)

        # Save chronology if allowed
        if self.saving_allowed:
            savepath = self.get_savepath(path, "chronology")
            chronology.serialize(savepath)
            print(f"Chronology saved to {savepath}")

        return chronology


    def create_dataset(self, path : str, chronology : Chronology, loader_type : Optional[Type[Loader]], dataset_kwargs : dict, loader_kwargs : dict) -> SeriesDataset:
        if loader_type is None:
            raise ValueError("Loader type must be specified to create a dataset.")
        
        # Create dataset
        structure_to_dataset_loader = loader_type(**loader_kwargs)
        dataset = SeriesDataset(chronology=chronology, struct_loader=structure_to_dataset_loader, **dataset_kwargs)

        # Save dataset if allowed
        if self.saving_allowed:
            savepath = self.get_savepath(path, "series_dataset", loader_type.__name__)
            dataset.save(savepath)
            print(f"Dataset saved to {savepath}")

        return dataset
    
    def create_pickle(self, path : str, chronology : Chronology, loader_type : Optional[Type[Loader]], loader_kwargs : dict, loading_kwargs : dict) -> dict:
        if loader_type is None:
            raise ValueError("Loader type must be specified to create a pickle dataset.")
        
        # Create dataset
        structure_to_dataset_loader = loader_type(**loader_kwargs)
        dataset = structure_to_dataset_loader.load(chronology, **loading_kwargs)

        # Save dataset if allowed
        if self.saving_allowed:
            savepath = self.get_savepath(path, "pickle")
            with open(savepath, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Dataset saved to {savepath}")

        return dataset
    

    def load(self, 
             path : str, 
             data_type : Union[Type[Chronology], Type[SeriesDataset], Type[dict]], 
             loader_type : Optional[Type[Loader]] = None,
             chronology_kwargs : Optional[dict] = None,
             dataset_kwargs : Optional[dict] = None,
             loader_kwargs : Optional[dict] = None,
             loading_kwargs : Optional[dict] = None,
            ) -> Union[Chronology, SeriesDataset, dict]:
        savetype_str = self.get_savetype(data_type)
        savepath = self.get_savepath(path, savetype_str, loader_type.__name__ if loader_type is not None else None)

        if chronology_kwargs is None:
            chronology_kwargs = {}
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if loader_kwargs is None:
            loader_kwargs = {}
        if loading_kwargs is None:
            loading_kwargs = {}

        # Load data if it exists
        if not self.force_data_computation and os.path.exists(savepath):
            print(f"Loading dataset from {savepath}...")
            return self.save_types[savetype_str]["loading_func"](savepath)
        
        # Else, if the data is a series dataset, try to load the chronology and recompute the dataset
        if not self.force_data_computation and savetype_str != "chronology":
            savepath_chr = self.get_savepath(path, "chronology")

            if os.path.exists(savepath_chr):
                print(f"No dataset found in {self.save_folder}. Data structure found and loaded from {savepath_chr}. Re-computing the dataset from structure...")

                # Load chronology
                chronology = Chronology.deserialize(savepath_chr)

                # Create dataset
                dataset = self.create_dataset(path, chronology, loader_type, dataset_kwargs, loader_kwargs)
                return dataset
            
        # Else, create the data: chronology then dataset if required
        logs_str = "Data computation forced." if self.force_data_computation else f"No data found in {self.save_folder}."
        print(f"{logs_str}. Generating chronology...")
        chronology = self.create_chronology(path, chronology_kwargs)

        if savetype_str == "chronology":
            return chronology

        print("Generating dataset...")
        if savetype_str == "series_dataset":
            dataset = self.create_dataset(path, chronology, loader_type, dataset_kwargs, loader_kwargs)
        elif savetype_str == "pickle":
            dataset = self.create_pickle(path, chronology, loader_type, loader_kwargs, loading_kwargs)
        else:
            raise ValueError(f"Data type {savetype_str} not supported.")
        return dataset


    @classmethod
    def load_data(cls, 
                path : str,
                data_type : Union[Type[Chronology], Type[SeriesDataset], Type[dict]],
                loader_type : Optional[Type[Loader]] = None,
                chronology_kwargs : Optional[dict] = None,
                dataset_kwargs : Optional[dict] = None,
                loader_kwargs : Optional[dict] = None,
                loading_kwargs : Optional[dict] = None,
                save_folder : str = DATA_STRUCTURES_SAVE_FOLDER_DEFAULT,
                saving_allowed : bool = True,
                force_data_computation : bool = False,
                path_components_in_save_name : int = 2,
            ) -> Union[Chronology, SeriesDataset, dict]:
        datasave = cls(save_folder, saving_allowed, force_data_computation, path_components_in_save_name)
        return datasave.load(path, data_type, loader_type, chronology_kwargs, dataset_kwargs, loader_kwargs, loading_kwargs)


