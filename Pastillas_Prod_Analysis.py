import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb




datos = pd.read_csv('datos.csv')
datos.rename(columns={"PreC_VM": "V1", "CP_VM": "V2", "Peso_VM": "V3", "Peso_min": "V4", "Peso_max": "V5", 'Diametro_VM': 'V6', "Dureza_VM": "V7", 'Diametro_VM.1': 'V8', "friab": "Target"}, inplace=True)
datos.drop('Unnamed: 0', axis=1, inplace=True)
datos.sort_values(by='indice_lote', inplace=True)
datos['Target'] = datos['Target']*100
datos.reset_index(drop=True, inplace=True)

scaler = StandardScaler()
datos_std = datos.drop('indice_lote', axis=1)
datos_std = scaler.fit_transform(datos_std)
datos_std = pd.DataFrame(datos_std, columns=datos.columns[1:])

x = datos_std.drop(['Target'], axis=1)
y = datos['Target']

x_train_RF, x_test_RF, y_train_RF, y_test_RF = train_test_split(x, y, test_size=0.2, random_state=42)

Random_Forest = RandomForestRegressor(n_estimators=100, random_state=42)
Random_Forest.fit(x_train_RF, y_train_RF)

importance_RF = Random_Forest.feature_importances_
feature_importance_RF = pd.Series(importance_RF, index=x.columns).sort_values(ascending=False)

explainer_RF = shap.TreeExplainer(Random_Forest)
shap_values_RF = explainer_RF.shap_values(x_test_RF)

y_pred_RF = Random_Forest.predict(x_test_RF)

mse_RF = mean_squared_error(y_test_RF, y_pred_RF)
r2_RF = r2_score(y_test_RF, y_pred_RF)


x_train_NN, x_test_NN, y_train_NN, y_test_NN = train_test_split(x, y, test_size=0.2, random_state=42)
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(x_train_NN, y_train_NN)

y_pred_NN = mlp.predict(x_test_NN)
result_NN = permutation_importance(mlp, x_test_NN, y_test_NN, n_repeats=10, random_state=42)
importances_sorted_idx = np.argsort(result_NN.importances_mean)[::-1]
scaled_importances_NN = result_NN.importances_mean / result_NN.importances_mean.sum()

explainer_NN = shap.KernelExplainer(mlp.predict, x_train_NN)
shap_values_NN = explainer_NN.shap_values(x_test_NN)

mse_NN = mean_squared_error(y_test_NN, y_pred_NN)
r2_NN = r2_score(y_test_NN, y_pred_NN)

x_train_XG, x_test_XG, y_train_XG, y_test_XG = train_test_split(x, y, test_size=0.2, random_state=42)

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xg_reg.fit(x_train_XG, y_train_XG)

importances_XG = xg_reg.get_booster().get_score(importance_type='weight')

# Calculate the total importance
total_importance_XG = sum(importances_XG.values())

# Scale the importances
scaled_importances_XG = {feature: importance / total_importance_XG for feature, importance in importances_XG.items()}

# Sort the scaled importances in descending order
sorted_scaled_importances_XG = dict(sorted(scaled_importances_XG.items(), key=lambda item: item[1], reverse=True))

explainer_XG = shap.TreeExplainer(xg_reg)
shap_values_XG = explainer_XG.shap_values(x_test_XG)

y_pred_XG = xg_reg.predict(x_test_XG)
mse_XG = mean_squared_error(y_test_XG, y_pred_XG)
r2_XG = r2_score(y_test_XG, y_pred_XG)



r_2_values = [r2_RF, r2_NN, r2_XG]
norm_r2 = [(r2 - min(r_2_values)) / (max(r_2_values) - min(r_2_values)) for r2 in r_2_values]

scaled_importances_XG = pd.Series(scaled_importances_XG)

scaled_importances_NN = pd.Series(scaled_importances_NN, index = x.columns[importances_sorted_idx])

average_importance = np.average([feature_importance_RF, scaled_importances_NN, scaled_importances_XG], axis = 0, weights = norm_r2)
average_importance = pd.Series(average_importance, index = x.columns)
average_importance = average_importance.sort_values(ascending=False)

shap_values_models = [shap_values_RF, shap_values_NN, shap_values_XG]

mean_pos_shap = []
mean_neg_shap = []

for i in range(len(shap_values_models[0])): #iterating through data points
    pos_shap_values = []
    neg_shap_values = []
    for model_index in range(len(shap_values_models)): 
        shap_values = shap_values_models[model_index][i]  
        pos_shap_values.append(np.maximum(shap_values, 0)) # positive values for current data point and current model
        neg_shap_values.append(np.minimum(shap_values, 0)) # negative values for current data point and current model

    # Calculate weighted means for current data point 
    mean_pos_shap.append(np.average(pos_shap_values, axis=0, weights=norm_r2))
    mean_neg_shap.append(np.average(neg_shap_values, axis=0, weights=norm_r2))


# Convert to numpy arrays for easier handling
mean_pos_shap = np.array(mean_pos_shap).mean(axis=0)
mean_neg_shap = np.array(mean_neg_shap).mean(axis=0)

mean_pos_impacts_avg = pd.Series(mean_pos_shap, index = x.columns)
mean_neg_impacts_avg = pd.Series(mean_neg_shap, index = x.columns)

variables_to_export = {
    'average_importance': average_importance,
    'mean_pos_impacts_avg': mean_pos_impacts_avg,
    'mean_neg_impacts_avg': mean_neg_impacts_avg,
    'x_train_RF': x_train_RF,
    'Random_Forest': Random_Forest
}
