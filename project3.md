# Project 3
## Lucy Greenman

## Part 1
### Download the dataset charleston_ask.csv and import it into your PyCharm project workspace. Specify and train a model that designates the asking price as your target variable and beds, baths and area (in square feet) as your features. Train and test your target and features using a linear regression model.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
```


```python
lin_reg = LinearRegression()
```


```python
df = pd.read_csv('charleston_ask.csv')
X = np.array(df.iloc[:,1:4])
y = np.array(df.iloc[:,0])
```


```python
def DoKFold(model, X, y, k, standardize=False, random_state=146):
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

    return train_scores, test_scores
```


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10,False)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.020
    Testing: -0.038


### Describe how your model performed. What were the training and testing scores you produced? How many folds did you assign when partitioning your training and testing data? Interpret and assess your output.

The model performed very poorly. I used 10 folds, and the training set showed a correlation of 0.020, while the testing set showed a correlation of -0.038. Neither of these is a strong correlation at all, so the model is not going to be very useful. This could indicate that number of beds, number of baths, and square footage are just not great predictors of asking price of a house, but we can't be sure of that just yet. Standardizing the data might help the model to have greater predictive power.

## Part 2
### Now standardize your features (again beds, baths and area) prior to training and testing with a linear regression model (also again with asking price as your target).


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10,True)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.020
    Testing: -0.038


### Now how did your model perform? What were the training and testing scores you produced? How many folds did you assign when partitioning your training and testing data? Interpret and assess your output.

Even with the features standardized, the model still does not have training or testing scores at all close to 1. The training score for the standardized data was 0.020, and the testing score was -0.038, both unchanged from when the data was not standardized. I used 10 folds just like before, so any change in the training and testing scores (or lack thereof, in this case) is due to standardization. In other words, standardizing the data did not seem to improve the predictive power of the model. It sounds like our features might just not have very much explanatory power, but let's try another model.

## Part 3
### Then train your dataset with the asking price as your target using a Ridge regression model.


```python
a_range = np.linspace(0, 100, 100)
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=False)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    Training score for this value: 0.019
    Testing score for this value: -0.036



```python
a_range = np.linspace(0, 100, 100)
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    Training score for this value: 0.019
    Testing score for this value: -0.034


### Now how did your model perform? What were the training and testing scores you produced? Did you standardize the data? Interpret and assess your output.

I ran the regression twice, first non-standardized and then standardized. Neither model performed better than the linear regression, standardized or non-standardized. The non-standardized Ridge regression yielded a training score of 0.019 and a testing score of -0.036, while the standardized Ridge gave the same training score of 0.019 and a slightly higher testing score of -0.034. Again, 10 folds were used consistently, so the only change between these data and Parts 1 and 2 is the type of regression used. Overall, the Ridge regression does not seem to improve the predictive power of our model over the linear regression, no matter whether both or either has been standardized. It's really sounding like our features (beds, baths, and square footage) just aren't very well correlated with our target (asking price). Maybe they're better correlated with a different target, say, the price that the house actually sold for?

## Part 4
### Next, go back, train and test each of the three previous model types/specifications, but this time use the dataset charleston_act.csv (actual sale prices). How did each of these three models perform after using the dataset that replaced asking price with the actual sale price? What were the training and testing scores you produced? Interpret and assess your output.


```python
df2 = pd.read_csv('charleston_act.csv')
X = np.array(df2.iloc[:,1:4])
y = np.array(df2.iloc[:,0])
```

Non-standardized linear regression:


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10,False)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.004
    Testing: -0.062


Standardized linear regression:


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10,True)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.004
    Testing: -0.062


Non-standardized Ridge regression:


```python
a_range = np.linspace(0, 100, 100)
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=False)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    Training score for this value: 0.004
    Testing score for this value: -0.056


Standardized Ridge regression:


```python
a_range = np.linspace(0, 100, 100)
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    Training score for this value: 0.004
    Testing score for this value: -0.055


When I swapped out the asking price data for the actual price data, the predictive power of the model did not improve. The two linear regressions (both non-standardized and standardized) showed the same results: a training score of 0.004 and a testing score of -0.062. The two Ridge regressions (non-standardized and standardized) also showed quite similar results to each other: the non-standardized had a training score of 0.004 and a testing score of -0.056, while the standardized had a training score of 0.004 and a testing score of -0.055. This indicates that the features selected (number of bedrooms, number of bathrooms, and square footage) do not predict the actual sale price of the house much better than they predict the asking price (in fact, training scores were worse with this target).

## Part 5
### Go back and also add the variables that indicate the zip code where each individual home is located within Charleston County, South Carolina.


```python
X = np.array(df.iloc[:,1:])
y = np.array(df.iloc[:,0])
```

### Train and test each of the three previous model types/specifications. What was the predictive power of each model? Interpret and assess your output.

Non-standardized linear regression with zip data:


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.281
    Testing: 0.169


Standardized linear regression with zip data:


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10,True)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.281
    Testing: 0.169


Non-standardized Ridge regression with zip data:


```python
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=False)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=9.2426e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=8.53266e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=4.7919e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.84921e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=4.07107e-24): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=2.03196e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.54022e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=3.80067e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,


    Training score for this value: 0.275
    Testing score for this value: 0.174


Standardized Ridge regression with zip data:


```python
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=6.38262e-17): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=2.22587e-17): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,


    Training score for this value: 0.276
    Testing score for this value: 0.193


Adding the zip code data drastically improved the training and testing scores across all models. The non-standardized and standardized linear regressions both gave identical results, a training score of 0.281 and a testing score of 0.169. These are both drastically improved (closer to 1, indicating a stronger correlation) over the prior values of 0.004 and -0.062. The non-standardized Ridge regression gave a training score of 0.275 and a testing score of 0.174, both an order of magnitude (or two!) closer to 1 than the previous values of 0.004 and -0.056. Finally, the standardized Ridge regression gave the highest scores yet: 0.276 for training and 0.193 for testing. The zip code data is clearly a much better predictor of asking price for a house than beds, baths, and square footage are, so adding it to our model improved both training and testing scores. This makes sense; houses of similar size and value tend to be grouped in neighborhoods, where they have the same zip code. Additionally, the Ridge regression made an improvement over the linear regression, and standardizing the Ridge gave an added boost, so our first few attempts were on the right track, we just didn't have the right feature yet!

What happens if we use the actual sale price rather than the asking price, including the zip code data?


```python
X = np.array(df2.iloc[:,1:])
y = np.array(df2.iloc[:,0])
```

Non-standardized linear regression with zip data, actual price:


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.339
    Testing: 0.208


Standardized linear regression with zip data, actual price:


```python
train_scores, test_scores = DoKFold(lin_reg,X,y,10,True)

print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.339
    Testing: -1558100304765821919428608.000


Non-standardized Ridge regression with zip data, actual price:


```python
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=False)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.30406e-22): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=7.26045e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.49675e-22): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.11675e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.43588e-22): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=1.40718e-22): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=2.42658e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,
    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=6.91709e-23): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,


    Training score for this value: 0.332
    Testing score for this value: 0.218


Standardized Ridge regression with zip data, actual price:


```python
k = 10

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
print('Training score for this value: ' + format(avg_tr_score[idx],'.3f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.3f'))
```

    /opt/conda/lib/python3.8/site-packages/sklearn/linear_model/_ridge.py:147: LinAlgWarning: Ill-conditioned matrix (rcond=8.10845e-17): result may not be accurate.
      return linalg.solve(A, Xy, sym_pos=True,


    Training score for this value: 0.333
    Testing score for this value: 0.219


The scores improve! The non-standardized linear regression has the highest testing score, 0.339, but its training score is slightly slower, 0.208. The standardized Ridge regression has the highest testing value, 0.219, with a training value of 0.333. The non-standardized Ridge regression is not far behind, with a training score of 0.332 and a testing score of 0.218. The standardized linear regression, though, has a training score of 0.339 and a testing score of...-1558100304765821919428608.000? It seems like there's some sort of error affecting the testing data when it includes zip codes, but only when we run a standardized linear regression. What happens if we scale the features?


```python
# basic scale charleston actual price
c_act = pd.read_csv('charleston_act.csv')
c_act[['prices_scale']] = c_act[['prices']]/100000 # prices
c_act[['sqft_scale']] = c_act[['sqft']]/1000 # prices
X = c_act.drop(["prices_scale","prices","sqft"],axis = 1)
y = c_act["prices_scale"]
X.shape
X = X.to_numpy()
kf = KFold(n_splits = 10, shuffle=True)
train_scores=[]
test_scores=[]
for idxTrain,idxTest in kf.split(X):
    Xtrain = X[idxTrain,:]
    Xtest = X[idxTest,:]
    ytrain = y[idxTrain]
    ytest = y[idxTest]
    lin_reg.fit(Xtrain,ytrain)
    train_scores.append(lin_reg.score(Xtrain,ytrain))
    test_scores.append(lin_reg.score(Xtest,ytest))
print('Training: ' + format(np.mean(train_scores), '.3f'))
print('Testing: ' + format(np.mean(test_scores), '.3f'))
```

    Training: 0.139
    Testing: -609720675156613645991936.000


Yikes, it doesn't look like that fixed the issue. Let's investigate what's going on with those kfolds.


```python
for idxTrain,idxTest in kf.split(X):
    Xtrain = X[idxTrain,:]
    Xtest = X[idxTest,:]
    print(Xtest)
    ytrain = y[idxTrain]
    ytest = y[idxTest]
    print(ytest)
    lin_reg.fit(Xtrain,ytrain)
    train_scores.append(lin_reg.score(Xtrain,ytrain))
    test_scores.append(lin_reg.score(Xtest,ytest))
```

    [[4.    2.    0.    ... 0.    0.    2.777]
     [3.    2.    0.    ... 0.    0.    2.782]
     [4.    4.    0.    ... 0.    0.    2.7  ]
     ...
     [4.    3.    0.    ... 0.    0.    2.146]
     [3.    2.    0.    ... 0.    0.    1.578]
     [3.    3.    0.    ... 0.    0.    1.834]]
    0      8.870
    9      2.840
    36     3.700
    81     3.995
    82     3.620
           ...  
    625    2.150
    627    5.150
    628    5.650
    647    5.500
    652    3.800
    Name: prices_scale, Length: 66, dtype: float64
    [[1.    1.    0.    ... 0.    0.    0.647]
     [2.    1.    0.    ... 0.    0.    0.525]
     [2.    3.    0.    ... 0.    0.    1.025]
     ...
     [2.    2.    0.    ... 0.    0.    0.96 ]
     [4.    3.    0.    ... 0.    0.    2.158]
     [2.    3.    0.    ... 0.    0.    1.296]]
    10     2.7500
    17     6.0000
    37     2.3000
    39     1.9500
    41     2.4990
            ...  
    617    3.5500
    620    7.2000
    637    5.6516
    640    4.2000
    656    9.4000
    Name: prices_scale, Length: 66, dtype: float64
    [[2.    2.    0.    ... 0.    0.    0.99 ]
     [2.    2.    0.    ... 0.    0.    1.036]
     [2.    2.    0.    ... 0.    0.    1.06 ]
     ...
     [4.    3.    0.    ... 0.    0.    1.48 ]
     [3.    2.    0.    ... 0.    0.    1.261]
     [3.    3.    0.    ... 0.    0.    1.308]]
    4      2.20500
    11     2.25000
    13     2.33500
    15     2.25000
    30     3.70000
            ...   
    624    2.02000
    639    0.10000
    642    2.34000
    650    4.77000
    654    3.74815
    Name: prices_scale, Length: 66, dtype: float64
    [[4.    4.    0.    ... 0.    0.    2.7  ]
     [3.    2.    0.    ... 0.    0.    1.2  ]
     [2.    2.    0.    ... 0.    0.    1.187]
     ...
     [3.    2.    0.    ... 0.    0.    1.699]
     [3.    2.    0.    ... 0.    0.    1.169]
     [3.    3.    0.    ... 0.    0.    1.391]]
    1      4.090
    5      3.450
    16     1.895
    19     6.800
    22     3.100
           ...  
    607    6.800
    610    4.450
    630    5.950
    641    3.570
    651    4.699
    Name: prices_scale, Length: 66, dtype: float64
    [[2.    1.    0.    ... 0.    0.    0.96 ]
     [3.    3.    0.    ... 0.    0.    2.175]
     [4.    4.    0.    ... 0.    0.    2.611]
     ...
     [3.    3.    0.    ... 0.    0.    1.81 ]
     [3.    3.    0.    ... 0.    0.    1.628]
     [3.    3.    0.    ... 0.    0.    2.44 ]]
    14     2.3500
    20     6.6500
    21     7.0000
    26     8.3000
    28     6.1500
            ...  
    594    3.5399
    605    4.0900
    632    2.0200
    638    3.5500
    658    4.1500
    Name: prices_scale, Length: 66, dtype: float64
    [[2.    3.    0.    ... 0.    0.    1.025]
     [2.    1.    0.    ... 0.    0.    0.525]
     [5.    4.    0.    ... 0.    0.    3.75 ]
     ...
     [5.    4.    0.    ... 0.    0.    4.278]
     [4.    5.    0.    ... 0.    0.    3.826]
     [4.    4.    1.    ... 0.    0.    2.279]]
    2      2.620
    51     8.250
    54     2.675
    55     5.450
    71     4.650
           ...  
    599    2.940
    631    1.050
    645    9.600
    653    4.330
    657    2.000
    Name: prices_scale, Length: 66, dtype: float64
    [[3.    2.    0.    ... 0.    0.    2.187]
     [3.    2.    0.    ... 0.    0.    1.84 ]
     [3.    2.    0.    ... 0.    0.    1.632]
     ...
     [4.    4.    0.    ... 0.    0.    2.625]
     [4.    4.    1.    ... 0.    0.    3.647]
     [3.    3.    0.    ... 0.    0.    1.828]]
    3      3.9040
    8      3.0000
    12     3.1500
    24     5.7200
    34     2.7940
            ...  
    633    2.0600
    635    5.8000
    643    5.7000
    648    2.4325
    655    2.8700
    Name: prices_scale, Length: 66, dtype: float64
    [[3.    2.    0.    ... 0.    0.    1.635]
     [3.    1.    0.    ... 0.    0.    1.08 ]
     [2.    1.    0.    ... 0.    0.    0.572]
     ...
     [5.    4.    0.    ... 0.    0.    2.721]
     [3.    2.    0.    ... 0.    0.    1.548]
     [1.    1.    0.    ... 0.    0.    0.504]]
    6      3.8700
    7      2.8100
    18     6.0000
    25     6.4000
    29     4.1000
            ...  
    623    5.6500
    626    2.4500
    636    5.4164
    644    8.1000
    646    2.2500
    Name: prices_scale, Length: 66, dtype: float64
    [[3.    3.    0.    ... 0.    0.    1.44 ]
     [4.    4.    0.    ... 0.    0.    2.611]
     [3.    2.    0.    ... 0.    0.    1.152]
     ...
     [2.    2.    0.    ... 0.    0.    1.008]
     [1.    1.    0.    ... 0.    0.    0.777]
     [4.    4.    0.    ... 0.    0.    3.054]]
    31     2.74900
    57     4.65000
    59     2.02225
    64     5.30000
    73     8.25000
            ...   
    615    3.76000
    621    2.06000
    634    3.09000
    649    3.24000
    659    5.99500
    Name: prices_scale, Length: 66, dtype: float64
    [[3.    2.    0.    ... 0.    0.    1.152]
     [3.    2.    0.    ... 1.    0.    2.187]
     [3.    2.    0.    ... 0.    0.    1.632]
     ...
     [4.    3.    0.    ... 0.    0.    1.48 ]
     [5.    4.    0.    ... 0.    0.    4.278]
     [3.    3.    0.    ... 0.    0.    1.391]]
    23     2.75000
    38     2.30000
    46     5.93362
    50     1.65000
    61     6.23500
            ...   
    600    1.50100
    601    1.25000
    609    5.55000
    613    4.29000
    618    5.55000
    Name: prices_scale, Length: 66, dtype: float64


Well, the scaling did what it was supposed to; both the Xtest and ytest data are now on the same scale. But the training score did not improve. It could be that there's a formatting error in the original dataframe, or that some zip code contained an error that only arises when we try to standardize it. For now, I'm just sticking with analysis of the asking price data, where this issue does not occur.

## Part 6
### Finally, consider the model that produced the best results. Would you estimate this model as being overfit or underfit? If you were working for Zillow as their chief data scientist, what action would you recommend in order to improve the predictive power of the model that produced your best results from the approximately 700 observations (716 asking / 660 actual)?

A model's fitness is defined as its validity externally compared with internally. That is, how well the model fits the testing data compared with the training data. An overfit model has lower external validity (a lower testing score than training score), while an underfit model has lower internal validity (a lower training score than testing score). The model that produced the best results was the standardized Ridge regression including the zip code data, which had a training score of 0.276 and a testing score of 0.193. Because the training score exceeds the testing score, this model is overfit. Its predictive power could be improved by using additional data, so that a more representative range of values could be used in both training and testing. Adding additional features could also increase predictive power, just like adding the dimension of zip code data improved upon the initial regressions. Maybe including the name of the development in which a home is located would correlate well with its price, since developments are smaller than zip code areas and are usually fairly uniform in style and value. Proximity to schools could also have a bearing on price, as could the availability of parking, or proximity to a city center. The housing market at the time of sale could also have great bearing on both asking and sale price; when many homes are available, sellers may have to accept lower prices due to the competition they face. Overall, adding additional features and more data points to this analysis would likely yield the best results.


```python

```
