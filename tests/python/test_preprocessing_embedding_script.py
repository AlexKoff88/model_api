import math
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import requests

# Temporary WA
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools/model_tools/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "demos/common/python"))

from openvino.model_zoo.model_api.models import Detection, Model

IMAFE_FILE = tempfile.NamedTemporaryFile(suffix=".jpg").name


def download_image(save_path):
    URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/image1.jpg"
    if not os.path.exists(save_path):
        r = requests.get(URL, allow_redirects=True)
        open(save_path, "wb").write(r.content)


def compare_output_objects(ref, obj):
    if type(ref) != type(obj):
        raise RuntimeError(
            f"Type of reference and object are different."
            f"Reference: {ref}, object: {obj}"
        )
    if isinstance(ref, tuple):  # treat as classification results
        if len(ref) != len(obj):
            raise RuntimeError(
                f"Length of reference and object are different."
                f"Reference: {ref}, object: {obj}"
            )
        # Classes should be the same but scores can be different by a small margin
        return ref[0] == obj[0] and math.isclose(ref[2], ref[2], rel_tol=1e-03)
    if isinstance(ref, Detection):
        result = True
        for ref_coord, obj_coord in zip(ref.get_coords(), obj.get_coords()):
            result = result and math.isclose(ref_coord, obj_coord, rel_tol=1.0)
        result = result and math.isclose(ref.score, obj.score, rel_tol=1e-03)
        result = result and ref.id == obj.id
        return result
    if isinstance(ref, np.ndarray):  # treat as segmentation masks
        diff_ratio = np.count_nonzero(ref - obj) / ref.size
        return diff_ratio < 0.03  # number of different pixels should be less than 3%
    return False


def compare_model_outputs(references, objects):
    references_to_compare = references
    objects_to_compare = objects
    if not isinstance(references, list):
        references_to_compare = [references]
        objects_to_compare = [objects]

    if len(references_to_compare) != len(objects_to_compare):
        raise RuntimeError(
            f"Length of reference and object are different."
            f"Reference: {references_to_compare}, object: {objects_to_compare}"
        )

    result = True
    for ref, obj in zip(references_to_compare, objects_to_compare):
        result = compare_output_objects(ref, obj)
        if result is False:
            assert (
                f"Results with embedded preprocessing does not correspond to the results without preprocessing."
                f"Reference: {references_to_compare}, object: {objects_to_compare}"
            )
            break
    return result


def test_image_models(model_name):
    download_image(IMAFE_FILE)

    image = cv2.imread(IMAFE_FILE)
    if image is None:
        raise RuntimeError("Failed to read the image")

    iters = 1
    model = Model.create_model(model_name, model_type="detection")
    start_ref = time.time()
    for i in range(iters):
        ref_output = model(image)
    elapsed_ref = time.time() - start_ref
    print(f"W/o preprocessing: {elapsed_ref / iters}")

    model_w_preprocess = Model.create_model(
        model_name, configuration={"embed_preprocessing": True}
    )
    start = time.time()
    for i in range(iters):
        to_compare = model_w_preprocess(image)
    elapsed = time.time() - start
    print(f"W preprocessing: {elapsed / iters}")

    assert compare_model_outputs(ref_output, to_compare)


from openvino.model_zoo.model_api.models import (
    classification_models,
    detection_models,
    segmentation_models,
)

# test_image_models("yolo-v4-tf")
# test_image_models("resnet-18-pytorch")
# test_image_models("fastseg-small")
# test_image_models("face-detection-retail-0044")
# test_image_models("mobilenet-v3-large-1.0-224-tf")
# test_image_models("efficientnet-b0-pytorch")
# test_image_models("ssdlite_mobilenet_v2")
test_image_models("ssd_mobilenet_v1_fpn_coco")

# for model_name in classification_models:
#     print(f"Validate model: {model_name}")
#     try:
#         test_image_models(model_name)
#     except BaseException as e:
#         print(f"Failed to validate model: {model_name}")
#         print(e)

# for model_name in detection_models:
#     print(f"Validate model: {model_name}")
#     try:
#         test_image_models(model_name)
#     except BaseException as e:
#         print(f"Failed to validate model: {model_name}")
#         print(e)

# for model_name in segmentation_models:
#     print(f"Validate model: {model_name}")
#     try:
#         test_image_models(model_name)
#     except BaseException as e:
#         print(f"Failed to validate model: {model_name}")
#         print(e)
