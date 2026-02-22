import os
import pathlib
from typing import List, Dict
import re
import numpy as np
import shutil

class FileName():
    def __init__(self, full_path: pathlib.PurePath, full_name: str, name: str, extension: str):
        """Init a FileName object 

        Args:
            full_path (PurePath): full path to a file
            full_name (str): name with extension
            name (str): name without extension
            extension (str): extension of file
        """
        self.full_path = full_path
        self.full_name = full_name
        self.name = name
        self.extension = extension


class Misc():
    @staticmethod
    def get_all_files(root_path: str, ext_filter: List[str]=None) -> List[FileName]:
        """Get all files in both root_path and its sub-folders

        Args:
            root_path (str): path of root folder
            ext_filter (list of str, optional): allowed extension such as .mp3, .wav .etc. Defaults to None.

        Returns:
            FileName: a FileName object

        Yields:
            Iterator[FileName]: return one FileName object
        """
        file_paths = {}
        if((ext_filter is not None) and (not isinstance(ext_filter, list))):
            raise Exception('ext_filter must be a list')

        for path, subdirs, files in os.walk(root_path):
            for name in files:
                idx = name.find('.')
                if(idx == -1):
                    ext = ''
                    name_no_ext = name 
                else:
                    ext_idx = name.rindex('.')
                    name_no_ext = name[:ext_idx]
                    ext = name[ext_idx:]
                
                if ext_filter is not None:
                    if ext not in ext_filter or (ext == ''):
                        continue
                full_path = pathlib.PurePath(path, name)
                fileName = FileName(full_path, name, name_no_ext, ext)
                yield fileName


    @staticmethod
    def get_split_all_lines(path: str, split=' |\t|\n') -> List[List[str]]:
        split_lines = []
        with open(path, 'r') as inFile:
            for line in inFile:
                splitLine = re.split(split, line)            
                splitLine = list(filter(None, splitLine))
                split_lines.append(splitLine)

        return split_lines

    @staticmethod
    def get_file_name_without_extension(full_file_path: str) -> str:        
        full_path = pathlib.PurePath(full_file_path)
        return full_path.name

    @staticmethod 
    def join_path(path: str, name: str) -> str:
        return os.path.join(path, name)


    @staticmethod
    def sub_indices_interp(idx: np.ndarray, n_seg=9) -> np.ndarray:
        """Linear interpolation index. 

        Args:
            idx (np.ndarray): index array for interpolation.
            n_seg (int, optional): number of indices per segment (also including original index). Defaults to 9. 
                                   This means each segment will have 8 more idx in between two indices.

        Returns:
            np.ndarray: The final interpolated array
        """
        sub_indices = np.zeros((idx.size * n_seg), dtype=int)
        org_idx = np.zeros_like(idx)  
        current_idx = 0
        for i in range(1, idx.size):        
            start = idx[i-1]
            delta = idx[i] - start
            step = max(1, int(delta / n_seg))
            end = min(idx[i], start + n_seg * step)
            interp = np.arange(start, end, step)
            end_idx = current_idx + interp.size
            sub_indices[current_idx:end_idx] = interp
            current_idx = end_idx
            org_idx[i] = current_idx
        sub_indices[current_idx] = idx[-1]
        return sub_indices[:current_idx+1], org_idx
    
    @staticmethod
    def try_get_data(data_folder_path: str, file_name: str, extentions: List[str]=['.mat', '.edf']) -> str:
        """Try get an data file when its extension matches specified extentions

        Args:
            data_folder_path (str): Absolute folder path to data
            file_name (str): Searched data name
            extentions (List[str]): Allowed extentions

        Returns:
            str: full path to data file
        """
        for ext in extentions:
            file_path = os.path.join(data_folder_path, file_name)
            if(os.path.exists(file_path)):
                return file_path
        return None
