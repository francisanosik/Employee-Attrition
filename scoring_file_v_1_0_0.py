# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"Age": pd.Series([0], dtype="int64"), "Gender": pd.Series(["example_value"], dtype="object"), "Years at Company": pd.Series([0], dtype="int64"), "Job Role": pd.Series(["example_value"], dtype="object"), "Monthly Income": pd.Series([0], dtype="int64"), "Work-Life Balance": pd.Series(["example_value"], dtype="object"), "Job Satisfaction": pd.Series(["example_value"], dtype="object"), "Performance Rating": pd.Series(["example_value"], dtype="object"), "Number of Promotions": pd.Series([0], dtype="int64"), "Overtime": pd.Series(["example_value"], dtype="object"), "Distance from Home": pd.Series([0], dtype="int64"), "Education Level": pd.Series(["example_value"], dtype="object"), "Marital Status": pd.Series(["example_value"], dtype="object"), "Number of Dependents": pd.Series([0], dtype="int64"), "Job Level": pd.Series(["example_value"], dtype="object"), "Company Size": pd.Series(["example_value"], dtype="object"), "Company Tenure": pd.Series([0], dtype="int64"), "Remote Work": pd.Series(["example_value"], dtype="object"), "Leadership Opportunities": pd.Series(["example_value"], dtype="object"), "Innovation Opportunities": pd.Series(["example_value"], dtype="object"), "Company Reputation": pd.Series(["example_value"], dtype="object"), "Employee Recognition": pd.Series(["example_value"], dtype="object")})
output_sample = np.array(["example_value"])
method_sample = StandardPythonParameterType("predict")

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, pd.DataFrame):
            result = result.values
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
