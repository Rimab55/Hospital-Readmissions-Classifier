
<h1 id="tocheading">Table of Contents</h1>
<div id="toc"></div>


```javascript
%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')
```


    <IPython.core.display.Javascript object>


# Motivation

Hospital readmission after an admitted patient is discharged is a high priority for hospitals. In total Hospital Readmissions are one of the most costly episodes to treat, costing Medicare about 26 billion annually, with about 17 billion consisting of avoidable trips after discharge. 

The health care burden of hospitalized patients with diabetes (1 and 2) is substantial and only growing. As of today, around 9% of Americans have diabetes or prediabetes, according to a recent CDC report. 

A better understanding of the factors that lead to hospital readmissions could help decision makers understand potential ways to reduce early readmissions (within 30 days) and provide more efficient care.

## Data Sources

- Diabetes patients records from 130 hospitals for visits occuring in 1998-2008 (10 years). Data was obtained from UCI Machine Learning DB: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008

- ICD-9 Codes diagnosis codes were scraped from: http://www.icd9data.com/2008/Volume1/default.htm

## Classification problem

**We focused our project on attempting to predict whether a discharged diabetic patient will be readmitted into a hospital within 30 days (Target = 1)  or not (Target = 0).**

At first, it seems like there are no null values. In reality, there is unavailable information which is represented by a question mark('?') in the dataframe.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 101766 entries, 0 to 101765
    Data columns (total 50 columns):
    encounter_id                101766 non-null int64
    patient_nbr                 101766 non-null int64
    race                        101766 non-null object
    gender                      101766 non-null object
    age                         101766 non-null object
    weight                      101766 non-null object
    admission_type_id           101766 non-null int64
    discharge_disposition_id    101766 non-null int64
    admission_source_id         101766 non-null int64
    time_in_hospital            101766 non-null int64
    payer_code                  101766 non-null object
    medical_specialty           101766 non-null object
    num_lab_procedures          101766 non-null int64
    num_procedures              101766 non-null int64
    num_medications             101766 non-null int64
    number_outpatient           101766 non-null int64
    number_emergency            101766 non-null int64
    number_inpatient            101766 non-null int64
    diag_1                      101766 non-null object
    diag_2                      101766 non-null object
    diag_3                      101766 non-null object
    number_diagnoses            101766 non-null int64
    max_glu_serum               101766 non-null object
    A1Cresult                   101766 non-null object
    metformin                   101766 non-null object
    repaglinide                 101766 non-null object
    nateglinide                 101766 non-null object
    chlorpropamide              101766 non-null object
    glimepiride                 101766 non-null object
    acetohexamide               101766 non-null object
    glipizide                   101766 non-null object
    glyburide                   101766 non-null object
    tolbutamide                 101766 non-null object
    pioglitazone                101766 non-null object
    rosiglitazone               101766 non-null object
    acarbose                    101766 non-null object
    miglitol                    101766 non-null object
    troglitazone                101766 non-null object
    tolazamide                  101766 non-null object
    examide                     101766 non-null object
    citoglipton                 101766 non-null object
    insulin                     101766 non-null object
    glyburide-metformin         101766 non-null object
    glipizide-metformin         101766 non-null object
    glimepiride-pioglitazone    101766 non-null object
    metformin-rosiglitazone     101766 non-null object
    metformin-pioglitazone      101766 non-null object
    change                      101766 non-null object
    diabetesMed                 101766 non-null object
    readmitted                  101766 non-null object
    dtypes: int64(13), object(37)
    memory usage: 38.8+ MB


## Mapping Admission, Discharge & Admission Source Types

A separate .csv file contains mapping codes to their description. The file contains all mapping codes, including:
- admission type codes
- discharge type codes
- admission source codes


```python
discharge_type_map.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>discharge_disposition_id</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>Discharged to home</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>Discharged/transferred to another short term h...</td>
    </tr>
  </tbody>
</table>
</div>



# EDA & Feature engineering

- Removing discharge codes 11 (Expired), 13-14 (Hospice care), 19-21 (Expired) as they consider the patient is terminally ill and the probability of being readmitted is equal to 0. 

It's important to note that the same patients appear several times as they can be admitted more than once. We decide to treat each patient's visit independently and not consider information regarding earlier visits as additional information.

For some patients' visits, gender is unknown or invalid and their corresponding rows will be removed as there are only 3 in total.

Some columns do not contain useful information regarding a patients' health and are dropped from our data:

- Patient_nbr: contains patient identification number
- Payer_code: contains insurance information

Many columns contain '?', None as an answer. For the following columns, they will be replaced by an appropriate value:

- For weight, race, medical_specialty, value will be replaced by "Not specified".
- For max_glu_serum, A1Cresult, value should be interpreted as "Not Measured", according to https://www.hindawi.com/journals/bmri/2014/781670/tab1/

## Target Variable: Setting Early (within 30 days) readmission to '1' 

>We are trying to predict patient early readmissions (within 30 days), therefore we combine readmissions over 30 days in the same group as non readmitted patients. In this case, we have a binary classification problem and logistic regression can be used as a base model.

##  Webscraping ICD-9 Codes & their descriptions

> In health care, diagnosis codes are used as a tool to group and identify diseases, disorders, symptoms, poisonings, adverse effects of drugs and chemicals, injuries and other reasons for patient encounters. Diagnostic coding is the translation of written descriptions of diseases, illnesses and injuries into codes from a particular classification. In medical classification, diagnosis codes are used as part of the clinical coding process alongside intervention codes. International Statistical Classification of Diseases and Related Health Problems (ICD) is one of the most widely used classification systems for diagnosis coding as it allows comparability and use of mortality and morbidity data.

>ICD codes in this dataset refer to ICD-9 codes.


## Grouping drugs by drug class

>To reduce dimensionality, we grouped 16 medications into 6 drug families.

> biguanides : The term biguanide refers to a group of oral type 2 diabetes drugs that work by preventing the production of glucose in the liver, improving the body's sensitivity towards insulin and reducing the amount of sugar absorbed by the intestines.

> meglitinides : The glinides are a class of drug which have a similar response as sulphonylureas but act for a shorter time. Meglitinides are prescribed to be taken by people with type 2 diabetes within half an hour before eating. As the drugs act for a shorter period than sulphonylureas, the side effects of hypoglycemia and weight gain have a smaller likelihood.

> sulfonylureas : Sulphonylureas are the class of antidiabetic drug for type 2 diabetes that tends to include those drugs which end in ‘ide’. They work by increasing the amount of insulin the pancreas produces and increasing the working effectiveness of insulin. The mode of action of sulphonylureas means that hypoglycemia and weight gain can be relatively common side effects.

> thiazolidinediones : Thiazolidinediones, also known as glitazones, are a medication for type 2 diabetes which help to improve insulin sensitivity and have been found to help decrease triglyceride levels.

> alpha_glucosidase_inhibitors : Alpha-glucosidase inhibitors, such as acarbose (marketed as Precose or Glucobay) or miglitol (branded as Glyset) are drugs for type 2 diabetes which slow down the digestion of carbohydrates in the small intestine and therefore can help to reduce after meal blood sugar levels.

> insulins : Insulin is a hormone which helps to regulate blood sugar. A number of different types of insulin are available as medication, with some insulins acting for as long as a day and others acting for only a few hours.

source : https://www.diabetes.co.uk/diabetes-medication/

We have also classified our patients as either "on" or "off" each medication, by changing the values 'Down, Up, Steady' to True and values 'No' to False.

Moreover, we created new variable columns for each drug class family and indicated whether a patient takes any medication within each family of drugs.

## Medical specialties: Retaining 10 medical specialties

> To reduce dimensionality of medical specialties, we have grouped all the medical specialties who had less than 500 patients into one category called "Other".

## Correlation Heatmap

> There is a -0.75 correlation between admission source id and admission type id. There is no correlation over |0.8| between numerical variables, therefore they will all be included in the initial model as regressors.


```python
# Correlation heatmap
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")

plt.figure(figsize=(14,12))

sns.heatmap(df.corr(),mask=mask, annot= True, vmax=.3,cmap = 'coolwarm_r', square=True, fmt='.2f',
            linewidths=2, cbar_kws={"shrink": 0.75})
plt.title('Correllation between numerical features', size = 20)
plt.xticks(rotation=80, size = 12)
plt.yticks(size = 12)
```




    (array([ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5, 10.5,
            11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5]),
     <a list of 21 Text yticklabel objects>)




![png](readmissions_clean-formd_files/readmissions_clean-formd_31_1.png)


## Combining 3 diagnosis codes into separate indicator dummy variables

> Each patient has 3 different diagnosis (diag1_desc, diag2_desc, diag3_desc). We decided to create dummy variables to indicate whether a patient had a diagnosis or not (True or False). To do so, we created a function to identify which diseases were present for each patient by encoding each of disease as a dummy variable.

## Exploring distribution of numerical variables


```python
df[['num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses','time_in_hospital']].boxplot(figsize = (18,9))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x128755a58>




![png](readmissions_clean-formd_files/readmissions_clean-formd_35_1.png)


# Part 2: Dimensionality reduction with PCA limited to 6 components

To reduce dimensionality, we have opted for an observed variance ratio cutoff at 0.075 meaning that each component should explain at least .075 of the total variance in order to be included as a principal component.


```python
X_scale = ['time_in_hospital',
 'num_lab_procedures',
 'num_procedures',
 'num_medications',
 'number_outpatient',
 'number_emergency',
 'number_inpatient',
 'number_diagnoses']
```


```python
#Standard Scaler to scale our continuous variables 

scaler = StandardScaler()
scaled_for_x = scaler.fit(df1[X_scale])
x = scaled_for_x.transform(df1[X_scale]) 

#using PCA to reduce dimensionality of our continuous variables 
pca = PCA(n_components = 6)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5','PCA_6'])

#examine the variance explained by our Principal Components
print(pca.explained_variance_ratio_)

```

    [0.25388298 0.17179902 0.12493312 0.11906943 0.10244657 0.09167577]



```python
#add our new PCA columns to our dataframe
df1 = pd.concat([principalDf, df1], axis=1) 
#drop our old continuous variable columns
df1.drop(X_scale, axis = 1, inplace=True)
```

# Part 3: Preparing data for modeling
## Dummy Variables Encoding for Categorical Data

>  Label encoding for categorical data : 'age','race', 'weight', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id',
       'medical_specialty', 'max_glu_serum', 'A1Cresult'

# Class Imbalance: Target variable
## Undersampling y & Setting Training/Test sets for X and y

> The target variable identifying whether a patient was readmitted early (within 30 days) is imbalanced, 11% of patients were readmitted within 30 days.
To combat imbalanced data, we opted to **downsample** the 'No' category. This will allow us to achieve a balance ratio closer to 1:1 and improve our binary classification predictions. Furthermore, we have opted to train on 80% of the data and test on the remainder.


```python
X_resampled, y_resampled = RandomUnderSampler().fit_sample(X,y)
```


```python
print(y.value_counts()) #Previous original class distribution
X_resampled, y_resampled = RandomUnderSampler().fit_sample(X, y) 
print(pd.Series(y_resampled).value_counts()) #Preview synthetic sample class distribution
```

    0    79716
    1    10265
    Name: readmitted, dtype: int64
    1    10265
    0    10265
    dtype: int64



```python
#split our data into a test and train set
X_train, X_test, y_train, y_test = train_test_split(X_resampled,y_resampled,test_size = .20) 
```

# Part 4: Modeling and Tuning models

## Baseline Model: Logistic Regression


```python
create_roc_auc(model_log, 'Logistic Regression')
```

    AUC_test: 0.6605263189113457
    AUC_train: 0.6689954220895153



![png](readmissions_clean-formd_files/readmissions_clean-formd_48_1.png)


### Tuning Logistic Regression

Tuning logistic regression predictions with parameter grid:

> Results are similar to base model, best parameters are mostly the default values. Best parameters C = 1, penalty score based on L1 - Lasso. (default is L2), number of iterations = 100. **Lasso** shrinks some coefficients close to 0 or to 0 by penalizing the cost function.


```python
# Best parameters
dt_grid_search.best_params_
```




    {'C': 1, 'max_iter': 100, 'penalty': 'l1'}




```python
print('accuracy train score:', accuracy_score(y_train, y_train_hat))
print('accuracy test score:', accuracy_score(y_test, y_test_hat))
print('recall score for test:', recall_score(y_test, y_test_hat)) 
print('recall score for train:', recall_score(y_train, y_train_hat))
print('precision score for test:', precision_score(y_test, y_test_hat))
print('precision score for train:', precision_score(y_train, y_train_hat))
```

    accuracy train score: 0.6179371651242085
    accuracy test score: 0.6144666341938626
    recall score for test: 0.5356622998544396
    recall score for train: 0.5414431984397855
    precision score for test: 0.638150289017341
    precision score for train: 0.6386772106398274


## KNN:  K-Nearest Neighbors

> Using GridSearch and comparing K = 5 and K = 100, our KNN model is optimized using K = 100. Results are poor and computationally expensive to optimize. KNN is not ideal for the size of our data.


```python
print('accuracy train score:', accuracy_score(y_train, y_train_hat))
print('accuracy test score:', accuracy_score(y_test, y_test_hat))
print('recall score for test:', recall_score(y_test, y_test_hat)) 
print('recall score for train:', recall_score(y_train, y_train_hat))
print('precision score for test:', precision_score(y_test, y_test_hat))
print('precision score for train:', precision_score(y_train, y_train_hat))
```

    accuracy train score: 0.610874330248417
    accuracy test score: 0.596687773989284
    recall score for test: 0.47792333818534694
    recall score for train: 0.49378352023403216
    precision score for test: 0.6293929712460063
    precision score for train: 0.6441405628875815



```python
create_roc_auc(knn_best, 'KNN with K = 100')
```

    AUC_test: 0.6426384039841082
    AUC_train: 0.6620307760382745



![png](readmissions_clean-formd_files/readmissions_clean-formd_56_1.png)


## Random Forest Classifier

> After GridSearch, parameters used are:
- criterion = 'entropy'
- max_depth = 10
- n_estimators = 100
- min_sample_split = 2 (Default)

> Beside the 6 principal components, the most important features with Random Forest include:
- Number of inpatient visits of the patient in the year preceding the encounter
- Discharge disposition code # 22 : Discharged/transferred to another rehab fac including rehab units of a hospital
- Number of emergency visits of the patient within the year preceding the encounter
- Number of lab tests performed during the encounter
- Time in hospital (measured in days)
- Number of procedures (Number of procedures (other than lab tests) performed during the encounter)
- Discharge disposition code # 3 : Discharged/transferred to SNF (skilled nursing facility)


```python
grid_rf.best_params_
```




    {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 115}




```python
print('accuracy train score:', accuracy_score(y_train, y_train_hat))
print('accuracy test score:', accuracy_score(y_test, y_test_hat))
print('recall score for test:', recall_score(y_test, y_test_hat)) 
print('recall score for train:', recall_score(y_train, y_train_hat))
print('precision score for test:', precision_score(y_test, y_test_hat))
print('precision score for train:', precision_score(y_train, y_train_hat))
```

    accuracy train score: 0.6921578178275695
    accuracy test score: 0.6122747199220653
    recall score for test: 0.532265890344493
    recall score for train: 0.6129936616284739
    precision score for test: 0.6359420289855072
    precision score for train: 0.7277858176555716



```python
# Plot feature importance
plot_feature_importances(best_rf)
```


![png](readmissions_clean-formd_files/readmissions_clean-formd_60_0.png)



```python
create_roc_auc(best_rf, 'Random Forest Classifier')
```

    AUC_test: 0.6575183077505282
    AUC_train: 0.76929553976993



![png](readmissions_clean-formd_files/readmissions_clean-formd_61_1.png)


## Boosted Trees with XG Boost

Furthermore, we have opted to classify patients using boosted trees (XGBoost) and use gridsearch to identify optimal parameters.


```python
print('accuracy train score:', accuracy_score(y_train, y_train_hat))
print('accuracy test score:', accuracy_score(y_test, y_test_hat))
print('recall score for test:', recall_score(y_test, y_test_hat)) 
print('recall score for train:', recall_score(y_train, y_train_hat))
print('precision score for test:', precision_score(y_test, y_test_hat))
print('precision score for train:', precision_score(y_train, y_train_hat))
```

    accuracy train score: 0.6314539698002922
    accuracy test score: 0.6147101802240623
    recall score for test: 0.5482775351770985
    recall score for train: 0.568503169185763
    precision score for test: 0.6344750140370579
    precision score for train: 0.6498536993172634



```python
xg_grid.best_params_
```




    {'gamma': 7, 'learning_rate': 0.16, 'max_depth': 3, 'n_estimators': 100}




```python
create_roc_auc(xg_fit, 'XG Boost Classifier')
```

    AUC_test: 0.653918564468313
    AUC_train: 0.6831183396978033



![png](readmissions_clean-formd_files/readmissions_clean-formd_65_1.png)


## SGD - Stochastic gradient descent


```python
sgd = SGDClassifier(loss = 'modified_huber',penalty="l1", max_iter=600)
sgd.fit(X_train, y_train) 
```




    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
           eta0=0.0, fit_intercept=True, l1_ratio=0.15,
           learning_rate='optimal', loss='modified_huber', max_iter=600,
           n_iter=None, n_jobs=1, penalty='l1', power_t=0.5, random_state=None,
           shuffle=True, tol=None, verbose=0, warm_start=False)




```python
y_train_hat = sgd.predict(X_train)
y_test_hat = sgd.predict(X_test)
```


```python
print('accuracy train score:', accuracy_score(y_train, y_train_hat))
print('accuracy test score:', accuracy_score(y_test, y_test_hat))
print('recall score for test:', recall_score(y_test, y_test_hat)) 
print('recall score for train:', recall_score(y_train, y_train_hat))
print('precision score for test:', precision_score(y_test, y_test_hat))
print('precision score for train:', precision_score(y_train, y_train_hat))
```

    accuracy train score: 0.6155625913297613
    accuracy test score: 0.6078908913784705
    recall score for test: 0.5143134400776322
    recall score for train: 0.5199902486591906
    precision score for test: 0.6351108448172559
    precision score for train: 0.6422764227642277



```python
create_roc_auc(sgd, 'Stochastic Gradient Descent')
```

    AUC_test: 0.6460351693874719
    AUC_train: 0.6570682837047028



![png](readmissions_clean-formd_files/readmissions_clean-formd_70_1.png)


# Results and Model comparison

- Our Random Forest Classifier has the highest train score, but seems to slightly overfit the training data.
- XGBoost has an overall better area under the curve for our testing data.


```python
auc_dict = {'Model': ['Log.Regression - Base','Log.Regression - Base',
                      'KNN','KNN',
                      'Random Forest Classifier','Random Forest Classifier',
                      'XGBoost','XGBoost','SGD','SGD'], 
            'AUC': [0.66,0.64,0.62,0.58,0.77,0.66,0.68,0.66, 0.65,0.66],
            'type':['Train','Test','Train','Test','Train','Test','Train','Test','Train','Test']}

# Bar plot
plt.figure(figsize=(12,8))
auc_scores = pd.DataFrame(auc_dict, columns = auc_dict.keys())
ax = sns.barplot(x="Model", y="AUC", hue="type", data=auc_scores, color='deepskyblue')
plt.title('Comparison of AUC by Model',fontsize = 15)
plt.legend(loc="upper right")
```




    <matplotlib.legend.Legend at 0x133a2cf98>




![png](readmissions_clean-formd_files/readmissions_clean-formd_73_1.png)



```python
accuracy_dict = {'Model': ['Log.Regression - Base','Log.Regression - Base','KNN','KNN','Random Forest','Random Forest',
                      'XGBoost','XGBoost', 'SGD' , 'SGD'], 
            'Accuracy': [0.62,0.62,0.61,0.59,0.68,0.62,0.63,0.62,0.62,0.62],
            'type': ['Train','Test','Train','Test','Train','Test','Train','Test','Train','Test']}

# Bar plot
plt.figure(figsize=(12,8))
accu_scores = pd.DataFrame(accuracy_dict, columns = accuracy_dict.keys())
ax = sns.barplot(x="Model", y="Accuracy", hue="type", data=accu_scores, color='deepskyblue')
plt.title('Comparison of Accuracy by Model',fontsize = 15)
plt.legend(loc="upper right")

```




    <matplotlib.legend.Legend at 0x1305cc550>




![png](readmissions_clean-formd_files/readmissions_clean-formd_74_1.png)


## Metric choice: Recall
We chose recall (or sensitivity) as the most important metric for our classification problem. We want to identify as many patients who are at risk of being readmitted within 30 days, even if it puts us at risk of misidentify patients who won't be readmitted in reality. We want the model to be sensitive and capture as many patients as possible who are at risk of being readmitted within 30 days. That being said, we are fully aware that our optimized models can  return many "false positives", meaning that they are more likely to have a higher Type 1 Error.


```python
recall_dict = {'Model': ['Log.Regression - Base','Log.Regression - Base','KNN','KNN','Random Forest','Random Forest',
                      'XGBoost','XGBoost', 'SGD' , 'SGD'], 
            'Recall Score': [0.54,0.54,0.49,0.48,0.61,0.53,0.57,0.55,0.52,0.51],
            'type': ['Train','Test','Train','Test','Train','Test','Train','Test','Train','Test']}

# Bar plot
plt.figure(figsize=(12,8))
recall_scores = pd.DataFrame(recall_dict, columns = recall_dict.keys())
ax = sns.barplot(x="Model", y="Recall Score", hue="type", data=recall_scores, color='deepskyblue')
plt.title('Comparison of Recall by Model',fontsize = 15)
plt.legend(loc="upper right")

```




    <matplotlib.legend.Legend at 0x13740bc88>




![png](readmissions_clean-formd_files/readmissions_clean-formd_76_1.png)


# Conclusion & Next Steps

For our Classifier we considered the most important evaluation metric to be recall.
In order to help hospital administration effectively address the issue of Hospital Readmissions <30 days we want our model to be able to correctly predict as many of the relevant cases as possible. 
No real surprise that the number of previous inpatient visits proved to be the strongest predictor of readmission within 30 days. 
Another strong predictor were discharge dispositions #22 and # 3, which correspond to discharge to a rehab facility or skilled nursing facility, respectively.
Patients who were on medications that fell into the ‘biguanides’ group showed a propensity to readmission.


**Model improvements:** 
- Feature Engineering
- Other models such as Support Vector Machine
- Multiclass with three possible classes (<30, >30, No)

**Recommendations:**
The most important indicator of readmission is the number of previous inpatient admissions, which could speak to the idea that an individual’s lifestyle choices and not simply their ailments must be addressed before and after release. 
Releasing a patient for continued treatment and rehabilitation might seem proactive and safe, however our data shows that discharge to these facilities prove to have no positive impact on overall health improvement.
A final thought was that the number of lab procedures proved to be more predictive than the number of diagnoses when identifying a patient who was going to be readmitted. This could be because lab procedures indicate a complicated diagnoses, but it also highlights the inefficient and expensive approach healthcare facilities take when treating patients. 

