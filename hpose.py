import importlib
import os.path
import platform
import re
import subprocess
from typing import Any

import cv2
import numpy as np
from requests import get as get_url
from tflite_runtime.interpreter import Interpreter


def get_device() -> tuple:
    """Return (os_id, os_machine) tuple"""
    output = subprocess.check_output(
        ['lsb_release', '-i']).decode('utf-8')
    os_id = re.search(r'.*:(.*)', output).group(1).strip()
    return (os_id, platform.machine())


def is_coral_board() -> bool:
    os_id, os_machine = get_device()
    return os_id == 'Mendel' and os_machine == 'aarch64'


def make_interpreter(model_path: str) -> Any:
    """Create a TFLite interpreter for model."""
    if is_coral_board():
        # Force using pycoral on Coral Dev Board
        pycoral = importlib.import_module(  # ImportError
            'pycoral.utils.edgetpu')
        i = pycoral.make_interpreter(model_path)
    else:
        i = Interpreter(model_path)
    print(f'Using model from {model_path}')
    i.allocate_tensors()
    return i


class MovenetDetector:
    """MovenetDetector

    It wraps a tflite.Interpreter loaded 
    with a MoveNet model. 
    """

    def __init__(self, lightning: bool = True) -> None:
        """Create a MoveNet interpreter 

        :param bool lightning: use lightning variant, defaults to True
        """
        self.model_path = self.get_or_download_model(
            lightning)
        self.interpreter = make_interpreter(
            self.model_path)

        self._input_idx = self.__get_detail_by_idx('index')
        self._output_idx = self.__get_detail_by_idx(
            'index', is_input=False)
        self._input_height, self._input_width = self.__get_detail_by_idx(
            'shape')[1:3]
        self._input_dtype = self.__get_detail_by_idx('dtype')

    def __get_detail_by_idx(self, key: str, idx: int = 0,
                            is_input: bool = True) -> Any:
        if is_input:
            return self.interpreter.get_input_details()[idx][key]
        return self.interpreter.get_output_details()[idx][key]

    def _preinvoke(self, image: np.ndarray) -> np.ndarray:
        """Transform input into tensor."""
        height, width = (int(x) for x in image.shape[:2])
        scale = min(self._input_width/width,
                    self._input_height/height)
        height, width = int(height*scale), int(width*scale)
        retval = np.zeros((1, self._input_height,
                           self._input_width, 3))
        retval[0, :height, :width] = cv2.resize(
            image, (width, height),
            interpolation=cv2.INTER_AREA)
        return retval.astype(self._input_dtype)

    def _invoke(self, data_in: np.ndarray) -> np.ndarray:
        """Invoke interpreter."""
        self.interpreter.set_tensor(
            self._input_idx, data_in)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self._output_idx)

    def _postinvoke(self, data_out: np.ndarray,
                    ratio: float) -> list:
        """Transform output tensor data to a 
        list of keypoints (x, y, score)."""
        ret = []
        for y, x, score in data_out[0][0]:
            ret.append((int(x*ratio), int(y*ratio),
                       float(score)))
        return ret

    def run(self, image: np.ndarray) -> list:
        """Run detector over image and return result."""
        data_in = self._preinvoke(image)
        data_out = self._invoke(data_in)
        return self._postinvoke(
            data_out, ratio=max(image.shape[:2]))

    @staticmethod
    def get_or_download_model(lightning: bool = True) -> str:
        """Get, or download, MoveNet model file.
        If downloaded, it is saved in the modules path.
        If device is a Coral Dev Board, it is forced to 
        look up for the Coral modified version of MoveNet.

        :param bool lightning: use lightning variant, defaults to True
        """
        directory = os.path.dirname(
            os.path.abspath(__file__))
        variant = 'lightning' if lightning else 'thunder'
        if is_coral_board():
            filename = f'movenet_{variant}_coral.tflite'
            url = 'https://github.com/google-coral/' +\
                'test_data/raw/104342d2d3480b3e66203073' +\
                f'dac24f4e2dbb4c41/movenet_single_pose_{variant}' +\
                '_ptq_edgetpu.tflite'
        else:
            filename = f'movenet_{variant}.tflite'
            url = 'https://tfhub.dev/google/lite-model/' +\
                f'movenet/singlepose/{variant}/3?lite-' +\
                'format=tflite'

        local_uri = os.path.join(
            directory, filename)
        if os.path.exists(local_uri):
            return local_uri  # found

        # Download
        response = get_url(url)
        if response.status_code != 200:
            raise RuntimeError(
                f'Cannot download model from {url}: ' +
                f'status {response.status_code}')

        # Save
        with open(local_uri, 'wb') as fw:
            fw.write(response.content)
        return local_uri
