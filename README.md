# Breast Cancer prediction
:::

::: {.cell .markdown id="m7A9KmYdb5LL"}
## Introduction

In this exercise we\'ll work with the Wisconsin Breast Cancer Dataset
from the UCI machine learning repository. We\'ll predict whether a tumor
is malignant or benign based on two features: the mean radius of the
tumor (radius_mean) and its mean number of concave points (concave
points_mean).
:::

::: {.cell .markdown id="eZfYRPH2cJHD"}
![breast
cancer](vertopal_85cdf330c77b4d03b636cedb2979c42c/0f3bb09fa9ad7343f8b84aa40610b429d5ea0ae3.jpg)
:::

::: {.cell .code id="3nzWpCFJcW9P"}
``` python
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
```
:::

::: {.cell .markdown id="jpglRz8leXqQ"}
## Dataset
:::

::: {.cell .code id="80pEsJktcdEa"}
``` python
df_cancer = pd.read_csv('wisconsin_breast_cancer.csv')
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="j-OT-4EqgJUa" outputId="12410164-135f-4dfc-8c35-dc9bd0e6012f"}
``` python
print(df_cancer.head())
```

::: {.output .stream .stdout}
             id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \
    0    842302         M        17.99         10.38          122.80     1001.0   
    1    842517         M        20.57         17.77          132.90     1326.0   
    2  84300903         M        19.69         21.25          130.00     1203.0   
    3  84348301         M        11.42         20.38           77.58      386.1   
    4  84358402         M        20.29         14.34          135.10     1297.0   

       smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \
    0          0.11840           0.27760          0.3001              0.14710   
    1          0.08474           0.07864          0.0869              0.07017   
    2          0.10960           0.15990          0.1974              0.12790   
    3          0.14250           0.28390          0.2414              0.10520   
    4          0.10030           0.13280          0.1980              0.10430   

       ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \
    0  ...          17.33           184.60      2019.0            0.1622   
    1  ...          23.41           158.80      1956.0            0.1238   
    2  ...          25.53           152.50      1709.0            0.1444   
    3  ...          26.50            98.87       567.7            0.2098   
    4  ...          16.67           152.20      1575.0            0.1374   

       compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \
    0             0.6656           0.7119                0.2654          0.4601   
    1             0.1866           0.2416                0.1860          0.2750   
    2             0.4245           0.4504                0.2430          0.3613   
    3             0.8663           0.6869                0.2575          0.6638   
    4             0.2050           0.4000                0.1625          0.2364   

       fractal_dimension_worst  Unnamed: 32  
    0                  0.11890          NaN  
    1                  0.08902          NaN  
    2                  0.08758          NaN  
    3                  0.17300          NaN  
    4                  0.07678          NaN  

    [5 rows x 33 columns]
:::
:::

::: {.cell .code id="R6fY4ME3die1"}
``` python
mapping = {'M':1, 'B':0}
df_cancer['diagnosis'] = df_cancer['diagnosis'].map(mapping)
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="YeDakM64gWOe" outputId="8d76901f-fd6a-4774-818e-168ce15dacbc"}
``` python
df_cancer.isna().sum()
```

::: {.output .execute_result execution_count="48"}
    id                           0
    diagnosis                    0
    radius_mean                  0
    texture_mean                 0
    perimeter_mean               0
    area_mean                    0
    smoothness_mean              0
    compactness_mean             0
    concavity_mean               0
    concave points_mean          0
    symmetry_mean                0
    fractal_dimension_mean       0
    radius_se                    0
    texture_se                   0
    perimeter_se                 0
    area_se                      0
    smoothness_se                0
    compactness_se               0
    concavity_se                 0
    concave points_se            0
    symmetry_se                  0
    fractal_dimension_se         0
    radius_worst                 0
    texture_worst                0
    perimeter_worst              0
    area_worst                   0
    smoothness_worst             0
    compactness_worst            0
    concavity_worst              0
    concave points_worst         0
    symmetry_worst               0
    fractal_dimension_worst      0
    Unnamed: 32                569
    dtype: int64
:::
:::

::: {.cell .code id="TQWMfhsZgGwo"}
``` python
df_cancer = df_cancer.drop(['Unnamed: 32'], axis=1)
```
:::

::: {.cell .code id="PpexbGt2fMeJ"}
``` python
# X and y data
X = df_cancer.drop(['diagnosis'], axis=1)
y = df_cancer[['diagnosis']]
```
:::

::: {.cell .markdown id="beUbn7F6eTgv"}
## Train/Test split
:::

::: {.cell .code id="1e1GHXNWcuY6"}
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
:::

::: {.cell .markdown id="gu72iyZTebpo"}
## Train the tree model
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="lYxx3cIPeM_F" outputId="69e1d890-7337-4eb7-fa46-2b18c68dde1e"}
``` python
# Instantiate decisiontreeclassifier
dt = DecisionTreeClassifier(max_depth=6, random_state=123)

# Fit to the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
print(y_pred[0:5])
```

::: {.output .stream .stdout}
    [1 0 1 1 1]
:::
:::

::: {.cell .markdown id="O6JxdkMqh1y3"}
## Evaluate classification
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="zJCz1V-5exaB" outputId="03bee918-a124-4663-c7e2-84ca3aa8941c"}
``` python
# Compute test set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))
```

::: {.output .stream .stdout}
    Test set accuracy: 0.90
:::
:::

::: {.cell .markdown id="Ks85FjIZjvkN"}
## Using entropy criterion
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":75}" id="ve6XCRUAidas" outputId="d214f8ba-a27f-46d3-8671-e7c755d43577"}
``` python
# Instantiate model with entropy criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)

# Fit the model to the training set
dt_entropy.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="57"}
```{=html}
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;, max_depth=8, random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(criterion=&#x27;entropy&#x27;, max_depth=8, random_state=1)</pre></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown id="GH_1DonNkNF1"}
## Predict and score
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0tYCohgZkFJO" outputId="200714f2-af98-403f-9804-79586892b474"}
``` python
# predict X_test
y_pred = dt_entropy.predict(X_test)

# Evaluate accuracy
accuracy_entropy = accuracy_score(y_test, y_pred)

print(f'Accuract achieved by using entropy: {accuracy_entropy:.3f}')
```

::: {.output .stream .stdout}
    Accuract achieved by using entropy: 0.860
:::
:::

::: {.cell .code id="zkZ-SELQkoqO"}
``` python
```
:::
