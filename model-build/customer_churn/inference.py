import io
import logging
import os
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
from constants import constants
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> lightgbm.Booster:
    """Read model saved in model_dir and return a object of lightgbm model.

    Args:
        model_dir (str): directory that saves the model artifact.

    Returns:
        obj: lightgbm model.
    """
    try:
        return joblib.load(os.path.join(model_dir, "model.pkl"))
    except Exception:
        logging.exception("Failed to load model from checkpoint")
        raise


def transform_fn(task: lightgbm.Booster, input_data: Any, content_type: str, accept: str) -> np.array:
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        task (lightgbm.Booster): model loaded by model_fn.
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.

    Returns:
        obj: the serialized prediction result or a tuple of the form
            (response_data, content_type)
    """
    if content_type == constants.REQUEST_CONTENT_TYPE:
        data = pd.read_csv(io.StringIO(input_data), sep=",", header=None)
        data.columns = [f"feature_{x}" for x in range(data.shape[1])]
        try:
            model_output = task.predict(data, num_iteration=task.best_iteration)
            if model_output.ndim == 1:  # Binary classification
                # Converting it into a 2-dimensional array to keep it consistent with catboost and sklearn
                model_output = np.vstack((1.0 - model_output, model_output)).transpose()
            output = {constants.PROBABILITIES: model_output}
            if accept.endswith(constants.VERBOSE_EXTENSION):
                predicted_label = np.argmax(model_output, axis=1)
                output[constants.PREDICTED_LABEL] = predicted_label
                accept = accept.rstrip(constants.VERBOSE_EXTENSION)
            return encoder.encode(output, accept)
        except Exception:
            logging.exception("Failed to do transform")
            raise
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
