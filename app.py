import streamlit as st
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from Pastillas_Prod_Analysis import *


def plot_variable_importance(variable_name):
  """Plots the importance of a single variable as a horizontal bar plot.

  Args:
    variable_name: The name of the variable to plot.
  """
  importance = average_importance[variable_name]

  plt.figure(figsize=(8, 2))  # Adjust figure size as needed
  plt.barh([variable_name], [importance], color='lightblue')
  plt.xlabel('Importance')
  plt.xlim(0, 1)  # Adjust x-axis limit
  plt.title(f'{variable_name} Importance')
  plt.text(importance + 0.001, 0, str(round(importance*100, 3))+'%', va='center')  # Add value label
  #plt.show()
  return plt

def plot_partial_dependence(model, X, features):
    # Plot using PartialDependenceDisplay
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=features,
        grid_resolution=50
    )
    #plt.show()
    return plt

def interpretation(variable):
  
  importance = round(average_importance[variable]*100, 2)

  average_impact_pos = round(mean_pos_impacts_avg[variable]*100, 3)
  average_impact_neg = round(mean_neg_impacts_avg[variable]*100, 3)


  importance_inter = ''
  impact_inter = ''
  sugerencia = ''

  # Interpretation of importance
  if importance >= 15:
      
      importance_inter += f'If we aim to understand why the target variable varies among CURRENT batches, {importance}% of these variations can be explained by changes in {variable}.\n'
      importance_inter += f'That is, {variable} may or may not be a direct cause of changes in the target variable, but this indicator suggests that {variable} is fluctuating significantly between batches and samples.'
      importance_inter += f' And this has affected the variability of the target variable.'
      sugerencia += f'To reduce high variations in the target variable, it is suggested to implement stricter control over {variable}.'

  elif importance >= 10 and importance < 15:

      importance_inter += f'If we aim to understand why the target variable varies among CURRENT batches, {importance}% of these variations can be explained by changes in {variable}.\n'
      importance_inter += f'That is, {variable} may or may not be a direct cause of changes in the target variable, but this indicator suggests that {variable} is fluctuating between batches and samples.'
      importance_inter += f' And this has affected the variability of the target variable.'
      sugerencia += f'To reduce high variations in the target variable, maintaining stricter control over {variable} could help mitigate this variability.'

  else:
      importance_inter += f'The variable {variable} does not typically fluctuate significantly between batches and samples. Therefore, the model found a low importance value of {importance}%.'
      sugerencia += f'This variable is under control regarding the variability of the target variable. That is, with the current level of fluctuation in {variable}, the target variable does not change significantly.'

  # Interpretation of impact

  if (average_impact_pos > 2.5 or average_impact_neg < -2.5) and variable != 'V7':
      impact_inter += f'The variable {variable} has a significant effect on the target variable within a batch. For each unit increase in {variable}, the target variable increases by an average of {average_impact_pos}%.\n'
      impact_inter += f'Conversely, for each unit decrease in {variable}, the target variable decreases by an average of {average_impact_neg}%.'
      sugerencia += f'\nSince the impact of {variable} is pronounced on the target variable, it is suggested to reduce the value of {variable} to decrease the target variable.'

  elif variable == 'V7': # Only variable with negative correlation
      impact_inter += f'The variable {variable} has a significant effect on the target variable within a batch. For each unit increase in {variable}, the target variable decreases by an average of {-1*average_impact_pos}%.\n'
      impact_inter += f'Conversely, for each unit decrease in {variable}, the target variable increases by an average of {abs(average_impact_neg)}%.'
      sugerencia += f'\nSince the impact of {variable} is significant on the target variable, it is suggested to increase the value of {variable} to decrease the target variable.'

  elif average_impact_pos > 1 or average_impact_neg < -1:
      impact_inter += f'The variable {variable} has a notable effect on the target variable within a batch. For each unit increase in {variable}, the target variable increases by an average of {average_impact_pos}%.\n'
      impact_inter += f'Conversely, for each unit decrease in {variable}, the target variable decreases by an average of {average_impact_neg}%.'
      sugerencia += f'\nSince the impact of {variable} is notable on the target variable within a batch, reducing the value of {variable} may help decrease the target variable.'

  elif average_impact_pos > 0.7 or average_impact_neg < -0.7:
      impact_inter += f'The variable {variable} has a minor effect on the target variable within a batch. For each unit increase in {variable}, the target variable increases by an average of {average_impact_pos}%.\n'
      impact_inter += f'Conversely, for each unit decrease in {variable}, the target variable decreases by an average of {average_impact_neg}%.'
      sugerencia += f'\nFinally, since the impact of {variable} is not particularly significant on the target variable within a batch, reducing the value of {variable} is not recommended to decrease the target variable.'

  else: 
      impact_inter += f'The direct impact of the variable {variable} on the target variable is not significant ({average_impact_pos}% per unit increase and {average_impact_neg}% per unit decrease).'
      sugerencia += f'\nFinally, since the impact of {variable} is not particularly significant on the target variable within a batch, reducing the value of {variable} is not recommended to decrease the target variable.'

  return importance_inter, impact_inter, sugerencia

def variable_analysis(variable):

  importance_int_var, impact_int_var, suggest_var = interpretation(variable)

  st.title(f"{variable} Analysis")

  
  st.pyplot(plot_variable_importance(variable))

  st.write(f'{importance_int_var}')

  
  st.pyplot(plot_partial_dependence(Random_Forest, x_train_RF, [variable]))
  st.subheader("Interpretation:")

  st.write(f'{impact_int_var}\n')
  st.write(f'{suggest_var}')

st.title("Feature Analyis")

selected_variable = st.selectbox("Feature: ", x.columns)

if st.button("Analyze!"):
  variable_analysis(selected_variable)
