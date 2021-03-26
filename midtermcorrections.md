# Midterm Corrections
## Lucy Greenman

## Background

### Importing libraries and data


```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
import numpy as np
```


```python
data = fetch_california_housing(as_frame=True)
lin_reg = LinearRegression()
```


```python
X = np.array(data.data)
y = np.array(data.target)
```

### Defining DoKFold

### My answer:


```python
def DoKFold(model, X, y, k, standardize=False, random_state=146):
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    train_scores = []
    test_scores = []

    for idxTrain,idxTest in kf.split(X):
        Xtrain = X[idxTrain,:]
        Xtest = X[idxTest,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]
        #Xtrain = np.reshape(Xtrain,(-1, 1))
        #Xtest = np.reshape(Xtest,(-1, 1))
        lin_reg.fit(Xtrain,ytrain)
        train_scores.append(lin_reg.score(Xtrain,ytrain))
        test_scores.append(lin_reg.score(Xtest,ytest))

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

    return train_scores, test_scores
```

### In-class answer:


```python
import pandas as pd
def DoKFold(model, X, y, k, standardize=False, random_state=146):
    if standardize:
        from sklearn.preprocessing import StandardScaler as SS
        ss = SS()

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
   
    train_scores = []
    test_scores = []

    train_mse = []
    test_mse = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain,:]
        Xtest = X[idxTest,:]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if standardize:
            Xtrain = ss.fit_transform(Xtrain)
            Xtest = ss.transform(Xtest)

        model.fit(Xtrain, ytrain)

        train_scores.append(model.score(Xtrain, ytrain))
        test_scores.append(model.score(Xtest, ytest))

        ytrain_pred = model.predict(Xtrain)
        ytest_pred = model.predict(Xtest)

        train_mse.append(np.mean((ytrain - ytrain_pred)**2))
        test_mse.append(np.mean((ytest - ytest_pred)**2))
        
    return train_scores, test_scores, train_mse, test_mse
```

## Question 19

### My work:


```python
from sklearn.linear_model import Ridge

a_range = np.linspace(20, 30, 101)

k = 20

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    rid_reg = Ridge(alpha=a)
    train_scores,test_scores = DoKFold(rid_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
optarid = a_range[idx]
print('Optimal alpha value: ' + format(a_range[idx], '.3f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.5f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.5f'))
```

    Optimal alpha value: 25.800
    Training score for this value: 0.60629
    Testing score for this value: 0.60200


### Answer from class:


```python
from sklearn.linear_model import Ridge, Lasso
k = 20
rid_a_range = np.linspace(20,30,101)

rid_tr=[]
rid_te=[]
rid_tr_mse=[]
rid_te_mse=[]

for a in rid_a_range:
    mdl = Ridge(alpha=a)
    train, test, train_mse, test_mse = DoKFold(mdl,X,y,k,True)
    
    rid_tr.append(np.mean(train))
    rid_te.append(np.mean(test))
    rid_tr_mse.append(np.mean(train_mse))
    rid_te_mse.append(np.mean(test_mse))
    
idx = np.argmax(rid_te)
print(rid_a_range[idx], rid_tr[idx], rid_te[idx], rid_tr_mse[idx], rid_te_mse[idx])
print('Optimal alpha value: ' + format(rid_a_range[idx], '.3f'))
print('Training score for this value: ' + format(rid_tr[idx],'.5f'))
print('Testing score for this value: ' + format(rid_te[idx], '.5f'))
```

    25.8 0.6062707743487987 0.6020111687418597 0.5242667528672735 0.528755670284223
    Optimal alpha value: 25.800
    Training score for this value: 0.60627
    Testing score for this value: 0.60201


### Correction/Reflection:

This question states: "Next, try Ridge regression. To save you some time, I've determined that you should look at 101 equally spaced values between 20 and 30 for alpha. Use the same settings for K-fold validation as in the previous question. For the optimal value of alpha in this range, what is the mean R2 value on the test folds?  Enter your answer to 5 decimal places, for example: 0.12345."

Like the answer given in class, I defined a_range as np.linspace(20, 30, 101). I set k = 20 and standardize = True. The other two parameters, random_state = 146 and shuffle = True, are defaults as defined in DoKFold. I then ran a Ridge regression and obtained an optimal alpha value of 25.8, a training score of 0.60629, and a testing score of 0.60200. Since the question asks for the mean R2 value on the test folds to 5 decimal places, I inputted 0.60200 as my answer.

As this comparison shows, the answer that I calculated is identical to the in-class answer to the 4th decimal place. We both found an alpha value of 25.000, but my training score was 0.60629 while the in-class answer shows 0.60627. My testing score was 0.60200, while the in-class answer was 0.60201. I'm not sure what exactly is causing this small discrepancy since we called just about exactly the same code, but I think the answers are close enough.

## Question 20

### My work:


```python
from sklearn.linear_model import Lasso

a_range = np.linspace(0.001,0.003,100)

k = 20

avg_tr_score=[]
avg_te_score=[]

for a in a_range:
    las_reg = Lasso(alpha=a)
    train_scores,test_scores = DoKFold(las_reg,X,y,k,standardize=True)
    avg_tr_score.append(np.mean(train_scores))
    avg_te_score.append(np.mean(test_scores))

idx = np.argmax(avg_te_score)
optalas = a_range[idx]
print('Optimal alpha value: ' + format(a_range[idx], '.5f'))
print('Training score for this value: ' + format(avg_tr_score[idx],'.5f'))
print('Testing score for this value: ' + format(avg_te_score[idx], '.5f'))
```

    Optimal alpha value: 0.00185
    Training score for this value: 0.60623
    Testing score for this value: 0.60206


### Answer from class:


```python
las_a_range = np.linspace(0.001,0.003,101)
las_tr=[]
las_te=[]
las_tr_mse=[]
las_te_mse=[]

for a in las_a_range:
    mdl = Lasso(alpha=a)
    train,test,train_mse,test_mse = DoKFold(mdl,X,y,k,True)
    
    las_tr.append(np.mean(train))
    las_te.append(np.mean(test))
    las_tr_mse.append(np.mean(train_mse))
    las_te_mse.append(np.mean(test_mse))
    
idx = np.argmax(las_te)
print(las_a_range[idx],las_tr[idx],las_te[idx],las_tr_mse[idx],las_te_mse[idx])
print('Optimal alpha value: ' + format(las_a_range[idx], '.5f'))
print('Training score for this value: ' + format(las_tr[idx],'.5f'))
print('Testing score for this value: ' + format(las_te[idx], '.5f'))
```

    0.00186 0.6061563795668892 0.6021329052825213 0.524419071473502 0.528600702331668
    Optimal alpha value: 0.00186
    Training score for this value: 0.60616
    Testing score for this value: 0.60213


### Correction/Reflection:

This question states: "Next, try Lasso regression.  Look at 101 equally spaced values between 0.001 and 0.003. Use the same settings for K-fold validation as in the previous 2 questions. For the optimal value of alpha in this range, what is the mean R2 value on the test folds? Enter you answer to 5 decimal places, for example: 0.12345."

Since this is the same question as the previous, just with the Lasso regression subbed in for the Ridge, I used the same code with that one tweak. I set a_range to (0.001, 0.003, 101), just like the answer given in class did. Then I ran my DoKFold with the default parameters of random_state = 146 and shuffle = True still in place, and I specified again that k = 20 and standardize = True. Thsi gave me an optimal alpha value of 0.00185, a training value of 0.60623, and a testing value of 0.60206. The question again asked for the mean testing score to 5 decimal places, so I inputted 0.60206.

The answer from class yields similar but not identical results: an alpha value of 0.00186, a training score of 0.60616, and a testing score of 0.60213. Again, we ran DoKFold with the exact same parameters, and our programs are written just about identically. I'm not sure what is causing this disconnect, but our answers are still the same to three or four decimal places.

## Question 24

### My work:


```python
from sklearn.preprocessing import StandardScaler as SS
ss = SS()
Xs = ss.fit_transform(X)
```


```python
a_range = np.linspace(0.001,0.003,101)

k = 20

MSEs=[]

for a in a_range:
    las_reg = Lasso(alpha=a)
    las_reg.fit(Xs,y)
    y_pred = las_reg.predict(Xs)
    prices = y
    predicted = y_pred
    summation = 0
    n = len(predicted)
    for i in range (0,n):
        difference = prices[i] - predicted[i]
        squared_difference = difference**2
        summation = summation + squared_difference
    MSE = summation/n
    MSEs.append(MSE)

print(min(MSEs))
idx = np.argmin(MSEs)
print('Optimal alpha value: ' + format(a_range[idx], '.6f'))
```

    0.5243769752107661
    Optimal alpha value: 0.001000



```python
las_reg = Lasso(alpha=0.00100)
las_reg.fit(Xs,y)
las_reg.score(Xs,y)
y_pred = las_reg.predict(Xs)
prices = y
predicted = y_pred

summation = 0
n = len(predicted)
for i in range (0,n):
    difference = prices[i] - predicted[i]
    squared_difference = difference**2
    summation = summation + squared_difference

MSE = summation/n
print ("The Mean Squared Error is: " , MSE)
```

    The Mean Squared Error is:  0.5243769752107661


### Answer from class:


```python
idx = np.argmin(las_te_mse)
print((las_a_range[idx],las_tr[idx],las_tr_mse[idx],las_te_mse[idx]))
```

    (0.00186, 0.6061563795668892, 0.524419071473502, 0.528600702331668)


### Correction/Reflection:

This question states: "If we had looked at MSE instead of R2 when doing our Lasso regression (question 20), what would we have determined the optimal value for alpha to be? Enter your answer to 5 decimal places, for example: 0.12345."

To get my answer, I did the calculation twice. First I used the traditional Lasso regression for loop to determine the optimal alpha value, and then I repeated the regression outside of the for loop using that alpha value to ensure that I would get the same MSE result, i.e. that there wasn't an issue in how my for loop was proceeding. I found an optimal alpha value of 0.00100 (interestingly enough, the same value that the answer in class found for Question 19) and a mean squared error of 0.52438.

The answer from class pulled out np.argmin of the MSEs just like I did, but it isolated the MSEs from the testing data. I didn't realize that we still needed to run DoKFold for this problem, so I just did it on the whole dataset without splitting into training and testing groups. Even so, the in-class answer is 0.52860, just a couple of thousandths away from my answer.


```python

```
