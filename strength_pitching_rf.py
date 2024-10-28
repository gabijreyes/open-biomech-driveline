import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os
import re

#set up plot saving
PLOT_FOLDER = r"C:\Users\gabi.reyes\Documents\Misc\open-biomech-driveline\results\strength_pitching_rf"
os.makedirs(PLOT_FOLDER, exist_ok=True)

def create_valid_filename(s):
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    s = s.replace(' ', '_')
    s = s.replace('[', '').replace(']', '')
    return s

def save_plot(fig, filename):
    valid_filename = create_valid_filename(filename)
    filepath = os.path.join(PLOT_FOLDER, valid_filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved: {filepath}")

#load data
hp_df = pd.read_csv(r"C:\Users\gabi.reyes\Documents\Misc\open-biomech-driveline\data\hp_obp.csv")

# correlations
hp_numeric_cols = hp_df.select_dtypes(include=[np.number]).columns.tolist()
hp_numeric_cols = [col for col in hp_numeric_cols if col not in ['bat_speed_mph', 'pitch_speed_mph', 'body_weight_[lbs]']]

hp_correlation = hp_df[hp_numeric_cols + ['pitch_speed_mph']].corr()['pitch_speed_mph'].sort_values(ascending=False)
top_10_correlations = pd.DataFrame({
    'Feature': hp_correlation.iloc[1:21].index,
    'Correlation': hp_correlation.iloc[1:21].values
})

print("\nTop strength correlations with pitch speed:")
print(tabulate(top_10_correlations, headers='keys', tablefmt='pretty', floatfmt='.4f', showindex=False))

#random forest
date_columns = ['test_date', 'pitching_session_date', 'hitting_session_date']
for col in date_columns:
    hp_df[col] = pd.to_datetime(hp_df[col], errors='coerce')

le = LabelEncoder()
hp_df['playing_level'] = le.fit_transform(hp_df['playing_level'])

hp_df['bat_speed_mph_group'] = hp_df['bat_speed_mph_group'].map({'<65': 0, '65-70': 1, '70+': 2})
hp_df['pitch_speed_mph_group'] = hp_df['pitch_speed_mph_group'].map({'<65': 0, '65-70': 1, '70+': 2})
hp_df = hp_df.dropna(subset=['pitch_speed_mph'])

features = [col for col in hp_df.columns if col not in ['pitch_speed_mph', 'athlete_uid', 'pitch_speed_mph_group'] + date_columns]

X = hp_df[features]
y = hp_df['pitch_speed_mph']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

feature_importance = pd.DataFrame({'feature': features, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

#feature heatmap - correlation matrix
top_10_features = feature_importance['feature'].tolist()
top_features_df = hp_df[['pitch_speed_mph'] + top_10_features]
correlation_matrix = top_features_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Top 10 Features with Pitch Speed')
plt.tight_layout()
save_plot(plt.gcf(), 'correlation_heatmap.png')

# pair plot
pair_plot = sns.pairplot(top_features_df, height=2, aspect=1.5)
pair_plot.fig.suptitle('Pair Plot of Top 10 Features and Bat Speed', y=1.02)
plt.tight_layout()
save_plot(pair_plot.fig, 'pair_plot.png')

#linear regression for top 10 features
def linear_regression_analysis(feature, df):
    X = df[[feature]]
    y = df['pitch_speed_mph']
    
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    r_squared = lr_model.score(X, y)
    correlation = df[feature].corr(df['pitch_speed_mph'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=feature, y='pitch_speed_mph', data=df, scatter_kws={'color': 'blue','alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    ax.set_xlabel(feature)
    ax.set_ylabel('Pitch Speed (mph)')
    ax.set_title(f'Linear Regression: Pitch Speed vs {feature}')
    equation = f"Pitch Speed = {lr_model.intercept_:.2f} + {lr_model.coef_[0]:.2f} * {feature}"
    textstr = f"Linear Regression Equation:\n{equation}\n\nR-squared: {r_squared:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    plt.tight_layout()
    save_plot(fig, f'linear_regression_{feature}.png')
    
    return lr_model.intercept_, lr_model.coef_[0], r_squared, correlation, len(df)

df = hp_df[top_10_features + ['pitch_speed_mph']].copy()
df = df.dropna()
results = []

for feature in top_10_features:
    intercept, coefficient, r_squared, correlation, n_samples = linear_regression_analysis(feature, df)
    results.append({
        'Feature': feature,
        'Intercept': intercept,
        'Coefficient': coefficient,
        'R-squared': r_squared,
        'Correlation': correlation,
        'Samples': n_samples
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R-squared', ascending=False)
pd.set_option('display.float_format', '{:.4f}'.format)
print("\nLinear Regression Results for Top 10 Features:")
print(results_df)

plt.figure(figsize=(12, 6))
sns.barplot(x='R-squared', y='Feature', data=results_df)
plt.title('R-squared Values for Top 10 Features')
plt.tight_layout()
save_plot(plt.gcf(), 'r_squared_barplot.png')