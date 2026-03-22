import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv("Fifa21Rawdata.csv", low_memory=False)

data.info()

print('-------------------------')
print(data.shape)

print('-------------------------')
print(data.head())


columns_to_drop = ["ID", "Name", "LongName", "photoUrl", "playerUrl",
    "Club", "Joined", "Contract", "Loan Date End",
    "Best Position",
    "GK Diving", "GK Handling", "GK Kicking",
    "GK Positioning", "GK Reflexes"]

clean_data = data.drop(columns=columns_to_drop)

print('-------------------------')
clean_data.info()

print('-------------------------')
print(clean_data.shape)

print('-------------------------')
print(clean_data.head())

print('-------------------------')
print(clean_data.dtypes)

print('-------------------------')
print(clean_data['Nationality'].value_counts())

print('-------------------------')
#correlations = clean_data.corr(method = 'pearson')
numeric_data = clean_data.select_dtypes(include=['int64','float64'])
correlations = numeric_data.corr(method = 'pearson')

#correlations = clean_data.corr(numeric_only=True) for less code

print(correlations)

print('-------------------------')
clean_data = clean_data.rename(columns={'↓OVA' : 'OVA'})

print(clean_data.columns)


print('-------------------------')
correlations_1 = clean_data.corr(numeric_only=True)
print(correlations_1['OVA'].sort_values(ascending=False))

#we found an extremely strong correlation between OVA and BOV(Best Overall Value) but the BOV is derived from OVA so we will drop it
#we have a very low correlation between OVA and Acceleration, Sprint Speed, Balance and Goalkeeping so we will drop them
'''
clean_data.plot(kind='hist'
                , subplots=True
                , sharex=False
                , sharey=False
                , layout=(4,12))
clean_data.plot(kind='density'
                , subplots=True
                , sharex=False
                , sharey=False
                , layout=(4,12))
clean_data.plot(kind='box'
                , subplots=True
                , sharex=False
                , sharey=False
                , layout=(4,12)) 

sns.heatmap(correlations,
            #annot=True,
            cmap='coolwarm'
            #,fmt=".2f"
            )

plt.show() 
'''

print ('----------------------------')
print(clean_data.isna().sum())
print(clean_data.isnull().sum())


print(clean_data.isna())

print(clean_data['Hits'])
print(clean_data['Hits'].head())

print ('Simple----------------------------Imputer')
# we need to replace the missing values by the mean (using the imputer, exactly the SimpleImputer)

clean_data['Hits'] = clean_data['Hits'].astype(str) #convert type to str (original type is obj)

clean_data['Hits'] = clean_data['Hits'].apply(
    lambda x: float(x.replace('K',''))*1000 if 'K' in x else x
    )
#Hits has vals such as 12k we need to convert them and change the other vals that are missing
'''
A lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression.
The basic syntax for a lambda function is: lambda arguments : expression '''

clean_data['Hits'] = pd.to_numeric(clean_data['Hits'], errors='coerce') #convert to numeric


imputer = SimpleImputer(strategy='median')
#we choose median cause , mean may NOT be ideal : Distribution is highly skewed,Few players have very high values,Many have low values 
#there is other type of imputers such as IterativeImputer and KNNImputer but SimpleInputer works with mean,median,most_frequent or constant

clean_data[['Hits']] = imputer.fit_transform(clean_data[['Hits']])

print(clean_data['Hits'].dtype)
print(clean_data['Hits'].isna().sum())

'''Handling Missing Values in Hits is necessary or obligatory only if im going to keep the feature
else i just drop it .
My Goal : Predict OVA from player attributes
Then Hits is probably irrelevant noise. '''

print ('Feature----------------------------Scaling')

features_to_keep = ['Reactions', 'Base Stats', 'Composure', 'PAS', 'DRI',
    'POT', 'Total Stats', 'Power', 'PHY', 'Shot Power',
    'Vision', 'Age']

X = clean_data[features_to_keep]
Y = clean_data['OVA']



'''
Feature scaling is needed because:
-Machine learning algorithms (like KNN, SVM, Logistic Regression, Neural Networks)
calculate distances or use gradients. If features have different ranges, features
with larger values dominate the calculation.
-Standardization improves convergence speed for gradient-based optimization.
-Makes coefficients more interpretable when using linear models.

Example:
-Age ranges from 16–40
-Shot Power ranges from 10–100
=>If you don’t scale, Shot Power will dominate.
'''

'''For your FIFA dataset, Standardization is often preferred because the ranges of features
like Age, Reactions, Shot Power, PAS vary, and we want them to contribute equally. '''
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X) # Fit scaler on X and transform

#print(X_scaled.head()) => need to transform it to DataFrame

X_scaled = pd.DataFrame(X_scaled, columns=X.columns) # Convert back to DataFrame for readability

print(X_scaled.head())

print(X_scaled.describe())



print ('--------------Modeling--------------')

print('Split the data : ')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)



# Choose your models

'''Since it's regression : we will try Linear Regression (baseline model) ,
                           Random Forest (more powerful) ,
                           KNN Regressor '''

# Linear Regression
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# KNN Regressor
from sklearn.neighbors import KNeighborsRegressor

model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(X_train, y_train)

# Predictions
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_knn = model_knn.predict(X_test)

# Creation of an Evaluation Function
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(name, y_test, y_pred): #takes 3 inputs:name :just a label (string),y_test :the real values,y_pred :predictions from your model
    print(f"----- {name} -----")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
    
    print("R2  :", r2_score(y_test, y_pred))
    print()

print ('----------Compare All Models-----------')
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("KNN", y_test, y_pred_knn)

'''Interpret results :
Focus on:
    *R²(most important) : Closer to 1 → better
        Example: 0.92 = very good / 0.78 = okay
    *RMSE : Lower = better : Represents average prediction error '''

''' PRO Tips :
    models = {
    "Linear Regression": y_pred_lr,
    "Random Forest": y_pred_rf,
    "KNN": y_pred_knn }
    for name, pred in models.items():
        evaluate_model(name, y_test, pred) '''

# We can Plot model comparison (r2_score)
models = ["Linear Regression", "Random Forest", "KNN"]
r2_scores = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_rf),
    r2_score(y_test, y_pred_knn)
]

plt.bar(models, r2_scores)
plt.title("Model Comparison (R2 Score)")
plt.show()

''' The results show : Best Model: Random Forest

Highest R² = 0.962 → explains ~96% of variance

Lowest RMSE = 1.34 → most accurate predictions

Lowest MAE = 0.89 → smallest average error '''

''' RQ : Random Forest significantly outperforms Linear Regression and KNN,
indicating that the relationship between player attributes and OVA
is non-linear and better captured by ensemble methods. '''

print ('Tuning------------------KNN')
#Tuning models (best hyperparameters to choose)

#Tuning KNN
for k in range(1, 15):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"k={k}, R2={r2_score(y_test, y_pred)}")
# Best K is 11 : k=11, R2=0.953597587178584
best_knn = KNeighborsRegressor(n_neighbors=11)
best_knn.fit(X_train, y_train)

y_pred_knn_best = best_knn.predict(X_test)

evaluate_model("Tuned KNN", y_test, y_pred_knn_best) #good results but still lower than Random Forst Regressor

print ('Tuning------------------RFR')
#Tuning Random Forest Regressor Using GridSearchCV (there is other methods but this is the best)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
    #'min_samples_split': [2, 5],     (these hyperpara not as important as the first 2 , and its good to eliminate these 2 if the PC slow) 
    #'min_samples_leaf': [1, 2]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,                # cross-validation
    scoring='r2',
    n_jobs=1,           # use all CPU => Problem: dataset is big (~15k rows)/ Random Forest + GridSearch + CV = heavy computation /PC runs out of memory → kills processes => so replace -1 by 1 
    verbose=2
)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

y_pred_best = best_rf.predict(X_test)

evaluate_model("Tuned Random Forest", y_test, y_pred_best)

'''mprovement is small but meaningful:
R² improved from 0.9619 to 0.9621
RMSE decreased slightly

This is completely normal in ML

-> When a model is already very good → gains are small
-> This shows your model was already near optimal '''

'''Best Model: Random Forest Regressor (Tuned)
Best Parameters:
- n_estimators = 200
- max_depth = None'''

importance = pd.Series(best_rf.feature_importances_, index=X.columns)
importance = importance.sort_values()

importance.plot(kind='barh')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

# Why "Reactions" is dominant:Reactions reflects how quickly a player responds to game situations,which is a critical factor in overall performance.
# In FIFA logic: Fast decision-making = better player , Affects both attack and defense
# Importance of "Base Stats":Base Stats represent aggregated player abilities,which naturally contribute to the overall rating.
# Role of "Potential (POT)":Potential indicates a player's future capability,which influences the overall evaluation of the player.
'''Why other features are low : !!!!!
Many features have low importance because their information
is already captured within more influential variables such as
Reactions and Base Stats.

This is called:
''Feature redundancy''  '''


#Final Model Selection
final_model = best_rf
#Save the Model (Deployment Step) (can reuse model later)
import joblib

joblib.dump(final_model, "fifa_model.pkl")
joblib.dump(scaler, "scaler.pkl")

#Prediction Function
def predict_player_ova(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    input_scaled = scaler.transform(input_df)
    
    input_scaled = pd.DataFrame(input_scaled, columns=X.columns)
    
    prediction = final_model.predict(input_scaled)
    return prediction[0]

print ('Example---------------Prediction')
sample_player = [80, 400, 75, 70, 78, 85, 1800, 75, 70, 80, 72, 25]

predicted_ova = predict_player_ova(sample_player)

print("Predicted OVA:", predicted_ova)


#Feature Reduction (OFC based on feature importance that we did later)
important_features = ['Reactions', 'Base Stats', 'POT']

#Retrain Model
X_reduced = clean_data[important_features]
X_scaled_reduced = scaler.fit_transform(X_reduced)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_scaled_reduced, Y, test_size=0.2, random_state=42
)

model_rf_reduced = RandomForestRegressor(n_estimators=200, random_state=42)
model_rf_reduced.fit(X_train_r, y_train_r)

y_pred_reduced = model_rf_reduced.predict(X_test_r)

evaluate_model("Reduced Features RF", y_test_r, y_pred_reduced)

'''A feature reduction experiment was conducted using only the most
important features identified by the Random Forest model
(Reactions, Base Stats, and Potential).

However, the model performance decreased significantly, with the
R² score dropping from 0.962 to 0.859.

This indicates that although some features have higher importance,
the combination of multiple features provides complementary
information that improves prediction accuracy.

Therefore, retaining a richer feature set leads to better model
performance, highlighting the importance of feature interactions
in machine learning models. '''

#Final Visualization (How close predictions are to reality)
plt.figure()
plt.scatter(y_test, y_pred_best)
plt.xlabel("Actual OVA")
plt.ylabel("Predicted OVA")
plt.title("Actual vs Predicted OVA")
plt.show()
'''Points close to diagonal = good predictions'''
