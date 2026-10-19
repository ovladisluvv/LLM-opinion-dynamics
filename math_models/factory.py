import numpy as np

from math_models.common import MathModel
from math_models.degroot import DegrootModel
from math_models.friedkin_johnsen import FriedkinJohnsenModel

def create_math_model(name: str, weights: np.ndarray, params: dict | None = None) -> MathModel:
    """Create a supported model with explicit checks for its parameters."""
    params = dict(params or {})

    if name == "degroot":
        if params:
            raise ValueError("DeGroot does not accept model parameters")
        return DegrootModel(weights)

    if name == "friedkin_johnsen":
        for parameter in params:
            if parameter != "susceptibility":
                raise ValueError(f"Unknown Friedkin-Johnsen parameter: {parameter!r}")
        if "susceptibility" not in params:
            raise ValueError("Friedkin-Johnsen requires susceptibility")
        return FriedkinJohnsenModel(weights, susceptibility=params["susceptibility"])

    raise ValueError(f"Unknown math model: {name!r}. Supported models: ['degroot', 'friedkin_johnsen']")
