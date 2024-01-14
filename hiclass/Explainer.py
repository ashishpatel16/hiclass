"""Explainer API for explaining predictions using shapley values."""
import shap
import numpy as np
from copy import deepcopy
from hiclass import (
    LocalClassifierPerParentNode,
    LocalClassifierPerNode,
    LocalClassifierPerLevel,
)


class Explainer:
    """Explainer class for returning shap values for each of the three hierarchical classifiers."""

    def __init__(self, hierarchical_model, data=None, algorithm="auto", mode="linear"):
        """
        Initialize the SHAP explainer for a hierarchical model.

        :param hierarchical_model: The hierarchical classification model to explain.
        :param mode: Can be `tree`, `gradient`, `deep`, or `linear`
        """
        self.hierarchical_model = hierarchical_model
        self.explainers = {}  # To store a SHAP explainer for each node
        self.algorithm = algorithm
        self.mode = mode
        self.data = data

        if mode == "tree":
            self.explainer = shap.TreeExplainer
        elif mode == "gradient":
            self.explainer = shap.GradientExplainer
        elif mode == "deep":
            self.explainer = shap.DeepExplainer
        elif mode == "linear":
            self.explainer = shap.LinearExplainer
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Supported modes are 'tree', 'gradient', 'deep', and 'linear'."
            )

    def explain(self, X):
        """
        Generate SHAP values for each node in the hierarchy for the given data.

        :param X: Data for which to generate SHAP values.
        :return: A dictionary of SHAP values for each node.
        """
        if isinstance(self.hierarchical_model, LocalClassifierPerParentNode):
            return self._explain_lcppn(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerLevel):
            return self._explain_lcpl(X)
        elif isinstance(self.hierarchical_model, LocalClassifierPerNode):
            return self._explain_lcpl(X)
        else:
            raise ValueError(f"Invalid model: {self.hierarchical_model}.")

    def _explain_lcppn(self, X):
        shap_values_dict = {}
        # TODO: Use predictions to restrict traversal to only visited path while computing shap values

        parent_nodes = self.hierarchical_model._get_parents()
        for parent_node in parent_nodes:
            # Ignore the designated root node which appears twice
            if parent_node == self.hierarchical_model.root_:
                continue

            # Get the local classifier for the parent node
            local_classifier = self.hierarchical_model.hierarchy_.nodes[parent_node][
                "classifier"
            ]

            # Create a SHAP explainer for the local classifier
            if parent_node not in self.explainers:
                # Create explainer with train data
                local_explainer = deepcopy(self.explainer)(local_classifier, self.data)
                self.explainers[parent_node] = local_explainer
            else:
                local_explainer = self.explainers[parent_node]

            # Calculate SHAP values for the given sample X
            shap_values = np.array(local_explainer.shap_values(X))
            shap_values_dict[parent_node] = shap_values

        return shap_values_dict

    def _explain_lcpn(self, X):
        pass

    def _explain_lcpl(self, X):
        pass
