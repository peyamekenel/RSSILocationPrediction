import pandas as pd
import numpy as np

# Read the results and transpose to get metrics as columns
df = pd.read_csv('model_comparison_results.csv', index_col=0).transpose()

# Convert RMSE and MSE columns to float
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
df['MSE'] = pd.to_numeric(df['MSE'], errors='coerce')

# Display RMSE and MSE for each model
print('\nRMSE and MSE by Model:')
print(df[['RMSE', 'MSE']].round(4))

# Find best (lowest) values
best_rmse = df['RMSE'].min()
best_mse = df['MSE'].min()
best_rmse_model = df['RMSE'].idxmin()
best_mse_model = df['MSE'].idxmin()

print(f'\nBest RMSE: {best_rmse:.4f} ({best_rmse_model})')
print(f'Best MSE: {best_mse:.4f} ({best_mse_model})')

# Verify Random Forest performance
rf_metrics = df.loc['Random Forest', ['RMSE', 'MSE']]
print(f'\nRandom Forest Metrics:')
print(f'RMSE: {rf_metrics["RMSE"]:.4f}')
print(f'MSE: {rf_metrics["MSE"]:.4f}')

# Verify if Random Forest is the best performer
is_best_rmse = rf_metrics['RMSE'] == best_rmse
is_best_mse = rf_metrics['MSE'] == best_mse
print(f'\nVerification Results:')
print(f'Random Forest has lowest RMSE: {is_best_rmse}')
print(f'Random Forest has lowest MSE: {is_best_mse}')
