import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier 

df = pd.read_csv('accidents_2012_to_2014.csv')

# limiting the dataset to only the year 2014
df = df.loc[df['Year'] == 2014]
# dropping unnecessary columns with high cardinality
df = df.drop(columns=['Accident_Index', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Local_Authority_(District)', 'Local_Authority_(Highway)', 'Junction_Detail', 'LSOA_of_Accident_Location', 'Year'])
df = df.set_index('Date')
# renaming the columns to put in lower case
df = df.set_axis(['longitude', 'latitude', 'police_force', 'accident_severity', 'number_of_vehicles', 'number_of_casualties', 'day_of_week', 'time', '1st_road_class', '1st_road_number', 'road_type', 'speed_limit', 'junction_control', '2nd_road_class', '2nd_road_number', 'pedestrian_crossing_human_control', 'pedestrian_crossing_physical_facilities', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site', 'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident'], axis=1)
# replacing "Yes" and "No" with "1" and "0"
df = df.replace(['Yes'],1)
df = df.replace(['No'],0)
# for the 'time' column, I just got rid of the : to change it to a int64
df['time'] = df['time'].astype('str')
df['time'] = df['time'].str.replace(':', '', regex=False)
df['time'] = df['time'].astype('int64')

df['did_police_officer_attend_scene_of_accident'].value_counts()

df['did_police_officer_attend_scene_of_accident'].value_counts(normalize=True).plot(kind='bar')


# reshaping the data to have a more balanced target column
df3 = df.loc[df['did_police_officer_attend_scene_of_accident'] == 0]
df2 = df.loc[df['did_police_officer_attend_scene_of_accident'] == 1]
df2.shape


df2 = df2.head(45000)
df2.shape
df2.head()


df = pd.concat([df2, df3])

#achieved a more balanced target column
df['did_police_officer_attend_scene_of_accident'].value_counts(normalize=True)

# split data
target = 'did_police_officer_attend_scene_of_accident'
X,y = df.drop(columns = target), df[target]


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42)


# establish baseline accuracy
print('The baseline accuracy is ', y_train.value_counts(normalize=True).max())


y.value_counts(normalize=True).plot(kind='bar')


# build multiple models


# build XGBClassifier model
model_xgb = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    XGBClassifier(random_state=42,n_estimators=1000, n_jobs=-1) # learning_rate=0.1
)
model_xgb.fit(X_train,y_train)


# Check Metrics
print('Training Accuracy', model_xgb.score(X_train, y_train))
print('Validation Accuracy', model_xgb.score(X_val, y_val))


# make the classification_report to look at precision and recall
from sklearn.metrics import classification_report
model_cr = classification_report(y_val, model_xgb.predict(X_val))
print(model_cr)


from sklearn.inspection import permutation_importance
results = permutation_importance(model_xgb, X_val, y_val, random_state=42)
                                 
perms = {'mean': results['importances_mean'],
         'std':results['importances_std']}
permutation_importances = pd.DataFrame(perms, index= X_val.columns).sort_values('mean')
permutation_importances


# look at feature importances for XGBClassifier
importances = model_xgb.named_steps['xgbclassifier'].feature_importances_
feature_names = X_train.columns
feat_imp = pd.Series(data=importances, index=feature_names).sort_values()
feat_imp.tail(10).plot(kind='barh')
plt.xlabel('Gini Importance')
plt.ylabel('Feature')

# from pdpbox.pdp import pdp_isolate, pdp_plot, pdp_interact, pdp_interact_plot

# look at pdp_plots for difference features
# feature = 'police_force'

# isolate = pdp_isolate(
#     model_xgb,
#     dataset= X_val,
#     model_features=X_val.columns,
#     feature=feature
# )

# pdp_plot(isolate, feature_name = feature);


# plot the confusion matrix for XGBCLassifier
from sklearn.metrics import plot_confusion_matrix, classification_report

plot_confusion_matrix(
    model_xgb,
    X_val, 
    y_val,
    values_format='.0f',
    display_labels=['police came', 'police did not come']
);

# make the classification_report to look at precision and recall
print(classification_report(y_val,
                      model_xgb.predict(X_val),
                      target_names=['police came', 'police did not come']))


# make the GradientBoostingClassifier model
from sklearn.ensemble import GradientBoostingClassifier

model_gb = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    GradientBoostingClassifier(random_state=42, n_estimators=1000)
)

model_gb.fit(X_train,y_train);


print('Training Accuracy', model_gb.score(X_train, y_train))
print('Validation Accuracy', model_gb.score(X_val, y_val))


# plot the confusion matrix for radientBoostingClassifier
plot_confusion_matrix(
    model_gb,
    X_val, 
    y_val,
    values_format='.0f',
    display_labels=['police came', 'police did not come']
);


# look at feature importances for Gradient Boosting Classifier
importances = model_gb.named_steps['gradientboostingclassifier'].feature_importances_
feature_names = X_train.columns
feat_imp = pd.Series(data=importances, index=feature_names).sort_values()
feat_imp.tail(10).plot(kind='barh')
plt.xlabel('Gini Importance')
plt.ylabel('Feature')

# look at pdp_plots for difference features
# feature = 'number_of_casualties'

# isolate = pdp_isolate(
#     model_gb,
#     dataset= X_val,
#     model_features=X_val.columns,
#     feature=feature
# )

# pdp_plot(isolate, feature_name = feature); 


# feature = 'police_force'

# isolate = pdp_isolate(
#     model_gb,
#     dataset= X_val,
#     model_features=X_val.columns,
#     feature=feature
# )

# pdp_plot(isolate, feature_name = feature);


print('Validation Accuracy', model_xgb.score(X_val, y_val))
print('Validation Accuracy', model_gb.score(X_val, y_val))

from sklearn.linear_model import LogisticRegression, LinearRegression
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_absolute_error,  r2_score
from sklearn.inspection import permutation_importance 

model_perm = make_pipeline(
    OrdinalEncoder(),
    SimpleImputer(strategy='mean'),
    LogisticRegression(max_iter=1000)
)

model_perm.fit(X_train,y_train);

print('Training Accuracy:', model_perm.score(X_train, y_train)) 
print('Validation Accuracy:',model_perm.score(X_val, y_val) )

print("R-Squared:", r2_score(y_train, model_perm.predict(X_train)))
print("R-Squared:", r2_score(y_val, model_perm.predict(X_val)))

col = 'police_force'

X_val_perm = X_val.copy()
X_val_perm[col] = np.random.permutation(X_val_perm[col])

print('validation accuracy', model_perm.score(X_val_perm,y_val))

perm_imp = permutation_importance(model_perm, X_val, y_val, random_state=42)

data_perm = {'imp_mean':perm_imp['importances_mean'],
             'imp_std':perm_imp['importances_std']}

df_perm = pd.DataFrame(data_perm, index=X_val.columns).sort_values('imp_mean')

df_perm['imp_mean'].tail(10).plot(kind='barh') 

y_pred_proba = model_perm.predict_proba(X_val)

y_pred_proba

# feature = 'police_force'

# isolate = pdp_isolate(
#     model_log,
#     dataset= X_val, # USE YOUR VALIDATION DATA
#     model_features=X_val.columns,
#     feature=feature
# )

# pdp_plot(isolate, feature_name = feature);

# feature = 'police_force'

# isolate = pdp_isolate(
#     model_gb,
#     dataset= X_val,
#     model_features=X_val.columns,
#     feature=feature
# )

# pdp_plot(isolate, feature_name = feature);

