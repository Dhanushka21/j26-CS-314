"""SHAP explanations for the detector model (tree-based stand-in; swap for DeepExplainer if Component 2 is a CNN/LSTM)."""
from typing import Dict, List

import numpy as np
import pandas as pd
import shap


class Explainer:
    def __init__(self, model, top_k: int = 3):
        self.model = model
        self.names = list(model.feature_names_in_)
        self.classes = list(model.classes_)
        self.top_k = top_k
        self.explainer = shap.TreeExplainer(model)

    def _values_for_class(self, x: List[float], predicted_class: str) -> np.ndarray:
        frame = pd.DataFrame([x], columns=self.names)
        sv = self.explainer.shap_values(frame)
        ci = self.classes.index(predicted_class)
        if isinstance(sv, list):          # older shap: list of (n, features), one per class
            return np.asarray(sv[ci])[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # newer shap: (n, features, classes)
            return sv[0, :, ci]
        return sv[0]

    def top_features(self, x: List[float], predicted_class: str) -> List[Dict]:
        vals = self._values_for_class(x, predicted_class)
        order = np.argsort(-np.abs(vals))[: self.top_k]
        return [{"feature": self.names[i], "contribution": round(float(vals[i]), 4)} for i in order]

    def shap_magnitude(self, x: List[float], predicted_class: str) -> float:
        return float(np.abs(self._values_for_class(x, predicted_class)).sum())


def summarize(attack_type: str, top: List[Dict]) -> str:
    names = ", ".join(f["feature"] for f in top)
    return f"Flagged as {attack_type}, mainly influenced by: {names}."
