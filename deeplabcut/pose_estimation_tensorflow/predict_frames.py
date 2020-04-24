"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

"""
from typing import Iterable, Union, Optional
from os import PathLike





# Represents strings and any filesystem path-like type....
Pathy = Union[PathLike, str]

def analyze_frame_store(config_path: Pathy, frame_stores: Iterable[Pathy], predictor: Optional[str] = None,
                        save_as_csv: bool = False, multi_output_format: str = "default", destfolder: Optional[str]=None,
                        num_outputs: Optional[int] = None):
    pass