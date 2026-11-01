# Copyright 2025 Wasserstein Distortion authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# ========================================================================================
"""Implementation of Wasserstein Distortion in JAX."""

from .helpers import lowpass
from .pretrained_features import load_vgg16_model
from .pretrained_features import compute_vgg16_features
from .wasserstein import multi_wasserstein_distortion
from .wasserstein import wasserstein_distortion
from .wasserstein import vgg16_wasserstein_distortion
