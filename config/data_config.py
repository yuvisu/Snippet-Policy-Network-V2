from datetime import datetime
from dataclasses import dataclass


@dataclass
class DataConfig():

    sampling_rate = 500

    segmenter: str = "gamboa"
    
    # path
    root_dir: str = ""

    data_dir: str = "ECG"

    snippet_dir: str = "snippet"

    tmp_dir: str = "tmp"

    output_dir: str = "eTSC"

    dataset_name: str = "ptbxl"

    snippet_name: str = "christov_norm_1000.pickle"