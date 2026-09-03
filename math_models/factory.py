import inspect
import numpy as np

from math_models.common import MathModel
from math_models.degroot import DegrootModel
from math_models.friedkin_johnsen import FriedkinJohnsenModel

# New models are registered here. A model class takes the weight matrix as its first
# argument and its own parameters as keyword arguments (see MathModel)
MATH_MODELS: dict[str, type[MathModel]] = {
    "degroot": DegrootModel,
    "friedkin_johnsen": FriedkinJohnsenModel,
}


def create_math_model(name: str, weights: np.ndarray, params: dict | None = None) -> MathModel:
    """Instantiate a registered model by name, translating parameter mistakes into ValueError"""
    if name not in MATH_MODELS:
        raise ValueError(f"Unknown math model: {name!r}. Supported models: {sorted(MATH_MODELS)}")

    model_class = MATH_MODELS[name]
    params = dict(params or {})

    signature = inspect.signature(model_class.__init__)
    expected = [parameter for parameter in list(signature.parameters)[2:]]

    unexpected = sorted(set(params) - set(expected))
    if unexpected:
        raise ValueError(f"Math model {name!r} does not accept parameters {unexpected}. Expected: {expected}")

    missing = [
        parameter_name for parameter_name, parameter in list(signature.parameters.items())[2:]
        if parameter.default is inspect.Parameter.empty and parameter_name not in params
    ]
    if missing:
        raise ValueError(f"Math model {name!r} requires parameters {missing}")

    return model_class(weights, **params)
