# -*- coding: utf-8 -*-
"""
=========================================
Explaining Local Classifier Per Level
=========================================

A minimalist example showing how to use HiClass Explainer to obtain SHAP values of LCPL model.
A detailed summary of the Explainer class has been given at Algorithms Overview Section for :ref:`Hierarchical Explainability`.
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from hiclass import LocalClassifierPerLevel, Explainer

# Define data
x_train = np.array(
    [
        [40.7, 1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 5.0, 34.3],
        [39.2, 0.0, 2.0, 4.0, 1.0, 3.0, 1.0, 2.0, 34.1],
        [40.6, 0.0, 3.0, 1.0, 4.0, 5.0, 0.0, 6.0, 27.7],
        [36.5, 0.0, 3.0, 1.0, 2.0, 2.0, 0.0, 2.0, 39.9],
    ]
)
x_test = np.array([[35.5, 0.0, 1.0, 1.0, 3.0, 3.0, 0.0, 2.0, 37.5]])
y_train = np.array(
    [
        ["Gastrointestinal", "Norovirus", ""],
        ["Respiratory", "Covid", ""],
        ["Allergy", "External", "Bee Allergy"],
        ["Respiratory", "Cold", ""],
    ]
)

# Use random forest classifiers for every level
rfc = RandomForestClassifier()
lcpl = LocalClassifierPerLevel(local_classifier=rfc, replace_classifiers=False)

# Train local classifiers per level
lcpl.fit(x_train, y_train)

# Define Explainer
explainer = Explainer(lcpl, data=x_train, mode="tree")
explanations = explainer.explain(x_test)

# Print out the results
print(explanations)

# Define a level
level = 1

# Use .sel() method
level_1_explanations = explanations.sel(level=1)

print(level_1_explanations)

print(level_1_explanations['classes'].values)

level_1_data = explanations.sel(level=1)
# Извлекаем классы, соответствующие первому уровню иерархии
# Выбираем данные для первого уровня иерархии

classes_level_1 = level_1_data["classes"]
desired_class = 'Cold_Allergy::HiClass::Separator::External'
print(type(classes_level_1))

for a in classes_level_1:
    print(a.values)