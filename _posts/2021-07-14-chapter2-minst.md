---
layout: post
date: 2021-07-14 4:31
title:  "MINST Classification"
mood: happy
category:
- python
- machine learning
---
# Chapter 3: Classification
Notes for *Hands on Machine Learning with Scikit by Aurelien Geron*. This is on Chapter 3: Classification, where it explains the fundamentals on how Machine Learning identify one group from the other using statistics.

# Getting Started
The dataset used is the *MNIST*, which is a collection of handwritten numbers in the format of image pixels. The goal is to have the machine tell us what number is in each image.


```python
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
pd.options.display.max_rows = 1000; pd.options.display.max_columns = 100;
```


```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])



You can download a copy of the dataset from sklearn. This gets updated often, which is why it is different on the book.


```python
df, target = mnist["data"], mnist["target"]
```

There are 2 sets of data. *Data* contains the pixel information about the number image, which is what we will use to predict. *Target* are the answer keys, which tells us what the number is.


```python
print('data', df.shape) # I store the data info in df, which is short for dataframe.
print('target', target.shape)
```

    data (70000, 784)
    target (70000,)



```python
df.iloc[0].describe()
```




    count    784.000000
    mean      35.108418
    std       79.699674
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        0.000000
    max      255.000000
    Name: 0, dtype: float64



There are 70,000 numbers in the dataset and each number holds 784 pixels. Each pixel has a value between 0 to 255, which indicates the intensity of the black color.

## Split Train Test
Before we begin, we set aside a portion of the dataset into a *Test* set, which is what we use at the very end to validate our models. We use the *Train* for majority of our exploration and data engineering.


```python
train_df, test_df, train_target, test_target = df[:60000], df[60000:], target[:60000], target[60000:]
```

The MINST dataset is already dividied into Train and Test. The first 60k rows is the Train and the rest are the Test.


```python
import numpy as np
shuffle_index = np.random.RandomState(seed=42).permutation(60000)
train_df, train_target = train_df.iloc[shuffle_index], train_target.iloc[shuffle_index]
```

There is a chance that the numbers are in order in a way that it will affect our models. Like a new deck of cards, we would always shuffle it before using them. Here, we shuffle it using the index.

# Explore Data
Our data is prepapred. It is time to Explore, Clean and Machine Learning the shit out of it.

I don't restrict myself from exploring the data but I do keep everything organized. That is why I keep Exploring, Cleaning and Machine Learning code in separate sections in this order. This section is where we keep all our Exploring data.


```python
import matplotlib as mpl
import matplotlib.pyplot as plt
seven = train_df.loc[59963].values
```

The numbers will keep shuffling everytime you run the code. Better choose a number and remember the index so you can iterate over the same number. Here I choose the number 7, which has an index of 59963 as an example.


```python
seven_target = train_target.loc[59963] # Choose a number and remember the index because of the shuffle.
print('the chosen number is:', seven_target)
plt.figure(figsize=(2,2))
plt.imshow(seven.reshape(28, 28), cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.savefig('seven.png')
plt.close()
```

    the chosen number is: 7


![seven](https://raw.githubusercontent.com/angelocandari/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/seven.png)

Each row contains pixel information about a number. If it reshaped into 28 by 28 pixels and plotted in a graph, the image of the number will show. We have chosen the number seven and true enough the image is a seven.

# Clean Data
In this section, we Clean the data.


```python
import numpy as np
train_target = train_target.astype(np.uint8)
test_target = test_target.astype(np.uint8)
```

The values of each targets are strings. We want them as intigers.

# Machine Learning
In this section, we do our Machine Learning algorithms. We start with an easy algorithm, then we work our way up to the more complex ones.

## Stochastic Gradient Descent
The goal is to use the pixel information in each row to tell us what number it is. But before doing all numbers, we start with something simple, which is to determine if the number is seven or not (TRUE or FALSE). For our first model, we use SGD that is great at classifying large datasets.


```python
train_target_seven = (train_target == int(seven_target))
```


```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42) # Init. Call on SGD.
sgd_clf.fit(train_df, train_target_seven) # Fit. We give instructions to Machine.
sgd_clf.predict([seven]) # Predict. Ask machine if seven is the number seven.
```




    array([ True])



Pay attention here because this is how all Machine Learning Algorithm works:
- Init. We create an instance of the model.
- Fit. We give the machine the data and the answer keys for it to learn.
- Predict. Then the machine uses what it learned from Fit to check if the Machine can tell if variable *seven* is seven. In this case, it does and it outputs the value *True*.

Now, let us check if the Machine can tell if the rest of the numbers are seven and check the accuracy.


```python
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, train_df, train_target_seven, cv=3, scoring="accuracy")
```




    array([0.97725, 0.9779 , 0.97535])



*cross_val_score* splits the training data into 3 folds and uses 2 folds as training data and 1 fold as the test. It does this 3 times in different combination.

Results show that on average, the model is accurate around ~95% of the time. This is unusually high. Let us try another dumber model and check.


```python
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
```


```python
never_roll_clf = Never5Classifier()
cross_val_score(never_roll_clf, train_df, train_target_seven, cv=3, scoring="accuracy")
```




    array([0.89245, 0.89785, 0.89645])



Again, we get very high results. This is because ~90% of the numbers are not sevens. So, if the model is lazy and said that all numbers are not seven, then it would be correct 90% of the time and it would give it a accuracy score of 90%. This is why accuracy, as a measure of classification, sucks.

## Confusion Matrix
Instead just counting the correct predictions, Confusion Matrix also takes into account the wrong predictions. This creates balance in the way we evaluate our models.


```python
from sklearn.model_selection import cross_val_predict
train_pred_seven = cross_val_predict(sgd_clf, train_df, train_target_seven, cv=3)
```


```python
from sklearn.metrics import confusion_matrix
confusion_matrix(train_target_seven, train_pred_seven)
```




    array([[52997,   738],
           [  652,  5613]], dtype=int64)



![confusion_matrix](https://raw.githubusercontent.com/angelocandari/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/image1.svg)

**Precision** is the number of times the model predicted that the number was seven versus all the predictions that it thought was seven. **Recall** is the number of times the model predicted that the number was seven versus all the actual sevens that exists in the dataset.

Precision alone is not enough to evaluate a model. If say the model had only 1 True prediction, which happens to be a seven, then your precision is 100%. Recall is also not useful either by itself. It is possible to get a 100% recall if the model predicted that ALL numbers are sevens, which gives you a Recall of 100%.

There are some use cases where you would prioritize one metric over the other but, generally, you want to have both Precision and Recall to be both high.


```python
from sklearn.metrics import precision_score, recall_score
precision = precision_score(train_target_seven, train_pred_seven)
precision
```




    0.8837978271138404



For our model, it has a 90% precision score, which means that out 5,992 numbers that it has identified as the number seven, only 5,402 of those wer actually the number seven. This is pretty good.


```python
recall = recall_score(train_target_seven, train_pred_seven)
recall
```




    0.8959297685554669



But it has scored a recall of 86%, which means that the 5,402 correct predictions are only 86% of all actual sevens. It was not able to identify 863 that were also sevens.

## F1 score
The F1 score is a convinient measure that calculates the *harmonic mean* of precision and recall.


```python
from sklearn.metrics import f1_score
print('f1: ', f1_score(train_target_seven, train_pred_seven))
```

    f1:  0.8898224476854788


This is different from just averaging the 2 scores, instead it calculates it in a way that gives more weight to low values. There are tradeoffs between precision and recall. You cannot increase one metric without decreasing the other.

Increasing Precision is actually telling the model to make predictions that has the least wrong answers, even if it does not get all the sevens in the dataset. Conversely, if Recall is prioritized, then the model needs to make sure that it gets all sevens in the dataset even if it mistakenly predict some numbers to be seven.

## Threshold
To better understand the tradeoffs between precision and recall, we are going to tweak the threshold of the models. By default, SGD's threshold is at zero, which finds the optimized scores between the 2 metrics.


```python
target_score = sgd_clf.decision_function([seven])
target_score
```




    array([7866.93916507])



We use the *decision_function* method of sgd to get the score of how well the model identifies the number data as seven.


```python
threshold = 0
seven_pred = (target_score > threshold)
seven_pred
```




    array([ True])



If we leave the threshold at zero, then we get a True output.


```python
threshold = 8000
seven_pred = (target_score > threshold)
seven_pred
```




    array([False])



If we set the threshold at 8000, then we get an output of False. This means that to the machine, the number does not enough score to be identified as the number seven.


```python
target_scores = cross_val_predict(sgd_clf, train_df, train_target_seven, cv=3, method='decision_function')
```


```python
pd.DataFrame({
    'target': train_target,
    'answer': train_target_seven,
    'scores': target_scores
}).query('target==7').sort_values(by='scores', ascending=False).head()
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
      <th>target</th>
      <th>answer</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37665</th>
      <td>7</td>
      <td>True</td>
      <td>48163.256141</td>
    </tr>
    <tr>
      <th>23000</th>
      <td>7</td>
      <td>True</td>
      <td>43371.436501</td>
    </tr>
    <tr>
      <th>21514</th>
      <td>7</td>
      <td>True</td>
      <td>41118.024888</td>
    </tr>
    <tr>
      <th>13244</th>
      <td>7</td>
      <td>True</td>
      <td>40826.558944</td>
    </tr>
    <tr>
      <th>26417</th>
      <td>7</td>
      <td>True</td>
      <td>40338.274713</td>
    </tr>
  </tbody>
</table>
</div>



If the score is ran on all the numbers on our train set, we get the corresponding score. Here is a preview.


```python
from sklearn.metrics import precision_recall_curve
precisions, recalls, threshold = precision_recall_curve(train_target_seven, target_scores)
```


```python
f1_score(train_target_seven, train_pred_seven)
```




    0.8898224476854788




```python
plt.figure(figsize=(8,5))
sns.lineplot(x=threshold, y=precisions[:-1], label='precision')
sns.lineplot(x=threshold, y=recalls[:-1], label='recall')
plt.legend()
plt.savefig('PrecvsReca.svg')
plt.close()
```

![precision_vs_recall](https://raw.githubusercontent.com/angelocandari/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/PrecvsReca.svg)

Here, we get to see precision and recall against there threshold. It is at threshold zero that we see the 2 lines intersect. This is where we will find the optimized scores for both precision and recall.

## ROC Curve
We can adjusting threshold to get different combination of precision and recall. But if you just want to evaluate the strength of your model on how it can predict something, then you would look at something like the ROC curve.


```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(train_target_seven, target_scores)
```


```python
plt.figure(figsize=(8,5))
sns.lineplot(x=fpr, y=tpr, label='SGD')
sns.lineplot(x=[0,1], y=[0,1], color='grey', linestyle='--', label='Randomness')
plt.legend()
plt.savefig('ROC_curve.svg')
plt.close()
```

![roc_curve](https://raw.githubusercontent.com/angelocandari/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/ROC_curve.svg)

The ROC curve plots the False Positive Rate on the x-axis and the True Positive rate on the y-axis. The FPR is the number of times the model says the numbers are sevens but are actually were not over all the Positive predictions it made. While the TPR is just the same as Recall. The ROC only cares about the Predictions that the model make as being TRUE and gauges the effectiveness of that model as it would a binomial distribution, whereby it determines how often it will get the right answer (TPR) against how likely it is to get the wrong answer (FPR). It disregards the FALSE output or the NEGATIVE output the model.


```python
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
probas_forest = cross_val_predict(forest_clf, train_df, train_target_seven, cv=3, method='predict_proba')

target_scores_forest = probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(train_target_seven, target_scores_forest)
```


```python
plt.figure(figsize=(8,5))
g = sns.lineplot(x=fpr, y=tpr, label='SGD', color='steelblue')
g = sns.lineplot(x=fpr_forest, y=tpr_forest, label='Random Forest', color='orange')
g = sns.lineplot(x=[0,1], y=[0,1], linestyle='--', color='black')
plt.savefig('rocSVGvsForest.svg')
plt.close()
```

![roc_sgd_forest](https://raw.githubusercontent.com/angelocandari/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/rocSVGvsForest.svg)
Here, we train a Random Forest model and place it against the SGD model in the same ROC plot. How ROC curve works is that the model should be as high up on the top left corner of the graph as possible. Comparing the 2 models, we get to see that the Random Forest Model is much better classifier than the SGD.

The dashed diagonal line in the middle represents randomness. If the model gets closer to that line, it means that it is no better at classifying than a coin flip.


```python
from sklearn.metrics import roc_auc_score
print('SGD', roc_auc_score(train_target_seven, target_scores))
print('RandomForest', roc_auc_score(train_target_seven, target_scores_forest))
```

    SGD 0.989954367264912
    RandomForest 0.9981929410171149


To summarize the results of both models to just one number, it is often easy to just caculate the area under the ROC curve and compare those instead of plotting it on a graph. Here, Random Forest is truely better model than the SGD.

## Multiclass Classification


```python
sgd_clf.fit(train_df, train_target)
sgd_clf.predict([seven])
```




    array([7], dtype=uint8)




```python
seven_scores = sgd_clf.decision_function([seven])
```


```python
seven_scores
```




    array([[-30680.33084323, -13285.00601243,  -6158.77253929,
             -7122.94195392, -18767.48190138,  -5392.96787129,
            -29211.19870979,   9929.67761202,    221.57103558,
             -6950.30435848]])



The model trains each number, not just seven, against all other numbers. It then uses that information to detect what number you are trying to predict.


```python
np.argmax(seven_scores)
```




    7




```python
sgd_clf.classes_
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)




```python
sgd_clf.classes_[7]
```




    7



### Random Forest


```python
forest_clf.fit(train_df, train_target)
forest_clf.predict([seven])
```




    array([7], dtype=uint8)




```python
forest_clf.predict_proba([seven])
```




    array([[0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.96, 0.02, 0.01]])



Random Forest sees the number that you are trying to predict is the number seven. It did the same thing as the SGD where it takes all the numbers and train itself.


```python
cross_val_score(sgd_clf, train_df, train_target, cv=3, scoring='accuracy')
```




    array([0.8678 , 0.88195, 0.86965])



When we evaluate, the accuracy is around 86%-88% accurate.

### Standard Scaler


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```


```python
train_df_scaled = scaler.fit_transform(train_df.astype(np.float64))
```


```python
cross_val_score(sgd_clf, train_df_scaled, train_target, cv=3, scoring='accuracy')
```




    array([0.90425, 0.9031 , 0.8903 ])



Here we scale the data to get a better score.

# Error Analysis
Knowing the scores from the evaluation is like knowing the summary of how good or bad the model is. We wouldn't really get to know why or what the model is getting right or getting wrong.


```python
train_target_pred = cross_val_predict(sgd_clf, train_df_scaled, train_target, cv=3)
conf_mx = confusion_matrix(train_target, train_target_pred)
conf_mx
```




    array([[5577,    0,   19,    5,   10,   39,   33,    5,  234,    1],
           [   1, 6413,   44,   17,    4,   45,    4,    8,  198,    8],
           [  23,   32, 5249,   88,   71,   20,   62,   39,  366,    8],
           [  27,   20,  115, 5216,    0,  197,   24,   44,  425,   63],
           [   7,   14,   46,   11, 5216,    8,   35,   20,  331,  154],
           [  28,   19,   31,  141,   52, 4466,   78,   18,  523,   65],
           [  27,   16,   50,    2,   39,   87, 5564,    7,  126,    0],
           [  20,   11,   52,   23,   50,   10,    4, 5702,  190,  203],
           [  18,   60,   42,  101,    2,  120,   32,   12, 5425,   39],
           [  22,   21,   29,   62,  123,   35,    1,  170,  361, 5125]],
          dtype=int64)



Here we go back to the confusion matrix and get to unpack all the numbers and the corresponding results.


```python
sns.heatmap(conf_mx);
plt.savefig('conf_mx.svg')
plt.close()
```

![conf_mx](https://raw.githubusercontent.com/angelocandari/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/conf_mx.svg)
Visualizing the data into a heatmap, we get to see precision (column) and recall (rows) by each of the numbers. The diagonal represents True | True results or the True Positives, where the model predicted the correct number. Don't misinterpret the shade of the heatmap. The lighter the color does not necessarily mean that the model is more precise or more accurate. The matrix defines precision and accuracy of the model by the position of the values it predicts. The color or the scores within the matrix is just there to keep count.


```python
pd.DataFrame(conf_mx).style.bar()
```




<style  type="text/css" >
#T_b9d53_row0_col0,#T_b9d53_row1_col1,#T_b9d53_row2_col2,#T_b9d53_row3_col3,#T_b9d53_row4_col4,#T_b9d53_row5_col5,#T_b9d53_row6_col6,#T_b9d53_row7_col7,#T_b9d53_row8_col8,#T_b9d53_row9_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 100.0%, transparent 100.0%);
        }#T_b9d53_row0_col1,#T_b9d53_row0_col2,#T_b9d53_row0_col7,#T_b9d53_row1_col0,#T_b9d53_row3_col4,#T_b9d53_row4_col5,#T_b9d53_row6_col3,#T_b9d53_row6_col8,#T_b9d53_row6_col9,#T_b9d53_row9_col6{
            width:  10em;
             height:  80%;
        }#T_b9d53_row0_col3,#T_b9d53_row1_col4,#T_b9d53_row1_col6,#T_b9d53_row1_col7,#T_b9d53_row4_col0,#T_b9d53_row7_col6,#T_b9d53_row8_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.1%, transparent 0.1%);
        }#T_b9d53_row0_col4,#T_b9d53_row1_col9,#T_b9d53_row2_col9,#T_b9d53_row4_col1,#T_b9d53_row4_col3,#T_b9d53_row5_col2,#T_b9d53_row5_col7,#T_b9d53_row6_col1,#T_b9d53_row7_col1,#T_b9d53_row9_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.2%, transparent 0.2%);
        }#T_b9d53_row0_col5,#T_b9d53_row3_col7,#T_b9d53_row6_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.7%, transparent 0.7%);
        }#T_b9d53_row0_col6,#T_b9d53_row2_col7,#T_b9d53_row4_col6,#T_b9d53_row6_col2,#T_b9d53_row7_col2,#T_b9d53_row8_col6,#T_b9d53_row9_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.6%, transparent 0.6%);
        }#T_b9d53_row0_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.0%, transparent 2.0%);
        }#T_b9d53_row0_col9,#T_b9d53_row6_col7,#T_b9d53_row7_col5,#T_b9d53_row8_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.0%, transparent 0.0%);
        }#T_b9d53_row1_col2,#T_b9d53_row2_col1,#T_b9d53_row3_col0,#T_b9d53_row4_col2,#T_b9d53_row5_col0,#T_b9d53_row6_col0{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.5%, transparent 0.5%);
        }#T_b9d53_row1_col3,#T_b9d53_row2_col5,#T_b9d53_row3_col1,#T_b9d53_row4_col7,#T_b9d53_row5_col1,#T_b9d53_row7_col0,#T_b9d53_row8_col0,#T_b9d53_row9_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.3%, transparent 0.3%);
        }#T_b9d53_row1_col5,#T_b9d53_row8_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.8%, transparent 0.8%);
        }#T_b9d53_row1_col8,#T_b9d53_row2_col4,#T_b9d53_row5_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.4%, transparent 1.4%);
        }#T_b9d53_row2_col0,#T_b9d53_row3_col6,#T_b9d53_row7_col3,#T_b9d53_row8_col2,#T_b9d53_row9_col0{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.4%, transparent 0.4%);
        }#T_b9d53_row2_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.6%, transparent 1.6%);
        }#T_b9d53_row2_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.1%, transparent 1.1%);
        }#T_b9d53_row2_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 4.5%, transparent 4.5%);
        }#T_b9d53_row3_col2,#T_b9d53_row6_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.8%, transparent 1.8%);
        }#T_b9d53_row3_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 4.2%, transparent 4.2%);
        }#T_b9d53_row3_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 5.6%, transparent 5.6%);
        }#T_b9d53_row3_col9,#T_b9d53_row7_col8,#T_b9d53_row9_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.2%, transparent 1.2%);
        }#T_b9d53_row4_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.9%, transparent 3.9%);
        }#T_b9d53_row4_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.0%, transparent 3.0%);
        }#T_b9d53_row5_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.7%, transparent 2.7%);
        }#T_b9d53_row5_col4,#T_b9d53_row7_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.0%, transparent 1.0%);
        }#T_b9d53_row5_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 7.5%, transparent 7.5%);
        }#T_b9d53_row5_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.3%, transparent 1.3%);
        }#T_b9d53_row7_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 4.0%, transparent 4.0%);
        }#T_b9d53_row8_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.9%, transparent 0.9%);
        }#T_b9d53_row8_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.9%, transparent 1.9%);
        }#T_b9d53_row8_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.5%, transparent 2.5%);
        }#T_b9d53_row9_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.4%, transparent 2.4%);
        }#T_b9d53_row9_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.9%, transparent 2.9%);
        }#T_b9d53_row9_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 4.4%, transparent 4.4%);
        }</style><table id="T_b9d53_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b9d53_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_b9d53_row0_col0" class="data row0 col0" >5577</td>
                        <td id="T_b9d53_row0_col1" class="data row0 col1" >0</td>
                        <td id="T_b9d53_row0_col2" class="data row0 col2" >19</td>
                        <td id="T_b9d53_row0_col3" class="data row0 col3" >5</td>
                        <td id="T_b9d53_row0_col4" class="data row0 col4" >10</td>
                        <td id="T_b9d53_row0_col5" class="data row0 col5" >39</td>
                        <td id="T_b9d53_row0_col6" class="data row0 col6" >33</td>
                        <td id="T_b9d53_row0_col7" class="data row0 col7" >5</td>
                        <td id="T_b9d53_row0_col8" class="data row0 col8" >234</td>
                        <td id="T_b9d53_row0_col9" class="data row0 col9" >1</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_b9d53_row1_col0" class="data row1 col0" >1</td>
                        <td id="T_b9d53_row1_col1" class="data row1 col1" >6413</td>
                        <td id="T_b9d53_row1_col2" class="data row1 col2" >44</td>
                        <td id="T_b9d53_row1_col3" class="data row1 col3" >17</td>
                        <td id="T_b9d53_row1_col4" class="data row1 col4" >4</td>
                        <td id="T_b9d53_row1_col5" class="data row1 col5" >45</td>
                        <td id="T_b9d53_row1_col6" class="data row1 col6" >4</td>
                        <td id="T_b9d53_row1_col7" class="data row1 col7" >8</td>
                        <td id="T_b9d53_row1_col8" class="data row1 col8" >198</td>
                        <td id="T_b9d53_row1_col9" class="data row1 col9" >8</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_b9d53_row2_col0" class="data row2 col0" >23</td>
                        <td id="T_b9d53_row2_col1" class="data row2 col1" >32</td>
                        <td id="T_b9d53_row2_col2" class="data row2 col2" >5249</td>
                        <td id="T_b9d53_row2_col3" class="data row2 col3" >88</td>
                        <td id="T_b9d53_row2_col4" class="data row2 col4" >71</td>
                        <td id="T_b9d53_row2_col5" class="data row2 col5" >20</td>
                        <td id="T_b9d53_row2_col6" class="data row2 col6" >62</td>
                        <td id="T_b9d53_row2_col7" class="data row2 col7" >39</td>
                        <td id="T_b9d53_row2_col8" class="data row2 col8" >366</td>
                        <td id="T_b9d53_row2_col9" class="data row2 col9" >8</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_b9d53_row3_col0" class="data row3 col0" >27</td>
                        <td id="T_b9d53_row3_col1" class="data row3 col1" >20</td>
                        <td id="T_b9d53_row3_col2" class="data row3 col2" >115</td>
                        <td id="T_b9d53_row3_col3" class="data row3 col3" >5216</td>
                        <td id="T_b9d53_row3_col4" class="data row3 col4" >0</td>
                        <td id="T_b9d53_row3_col5" class="data row3 col5" >197</td>
                        <td id="T_b9d53_row3_col6" class="data row3 col6" >24</td>
                        <td id="T_b9d53_row3_col7" class="data row3 col7" >44</td>
                        <td id="T_b9d53_row3_col8" class="data row3 col8" >425</td>
                        <td id="T_b9d53_row3_col9" class="data row3 col9" >63</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_b9d53_row4_col0" class="data row4 col0" >7</td>
                        <td id="T_b9d53_row4_col1" class="data row4 col1" >14</td>
                        <td id="T_b9d53_row4_col2" class="data row4 col2" >46</td>
                        <td id="T_b9d53_row4_col3" class="data row4 col3" >11</td>
                        <td id="T_b9d53_row4_col4" class="data row4 col4" >5216</td>
                        <td id="T_b9d53_row4_col5" class="data row4 col5" >8</td>
                        <td id="T_b9d53_row4_col6" class="data row4 col6" >35</td>
                        <td id="T_b9d53_row4_col7" class="data row4 col7" >20</td>
                        <td id="T_b9d53_row4_col8" class="data row4 col8" >331</td>
                        <td id="T_b9d53_row4_col9" class="data row4 col9" >154</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_b9d53_row5_col0" class="data row5 col0" >28</td>
                        <td id="T_b9d53_row5_col1" class="data row5 col1" >19</td>
                        <td id="T_b9d53_row5_col2" class="data row5 col2" >31</td>
                        <td id="T_b9d53_row5_col3" class="data row5 col3" >141</td>
                        <td id="T_b9d53_row5_col4" class="data row5 col4" >52</td>
                        <td id="T_b9d53_row5_col5" class="data row5 col5" >4466</td>
                        <td id="T_b9d53_row5_col6" class="data row5 col6" >78</td>
                        <td id="T_b9d53_row5_col7" class="data row5 col7" >18</td>
                        <td id="T_b9d53_row5_col8" class="data row5 col8" >523</td>
                        <td id="T_b9d53_row5_col9" class="data row5 col9" >65</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_b9d53_row6_col0" class="data row6 col0" >27</td>
                        <td id="T_b9d53_row6_col1" class="data row6 col1" >16</td>
                        <td id="T_b9d53_row6_col2" class="data row6 col2" >50</td>
                        <td id="T_b9d53_row6_col3" class="data row6 col3" >2</td>
                        <td id="T_b9d53_row6_col4" class="data row6 col4" >39</td>
                        <td id="T_b9d53_row6_col5" class="data row6 col5" >87</td>
                        <td id="T_b9d53_row6_col6" class="data row6 col6" >5564</td>
                        <td id="T_b9d53_row6_col7" class="data row6 col7" >7</td>
                        <td id="T_b9d53_row6_col8" class="data row6 col8" >126</td>
                        <td id="T_b9d53_row6_col9" class="data row6 col9" >0</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_b9d53_row7_col0" class="data row7 col0" >20</td>
                        <td id="T_b9d53_row7_col1" class="data row7 col1" >11</td>
                        <td id="T_b9d53_row7_col2" class="data row7 col2" >52</td>
                        <td id="T_b9d53_row7_col3" class="data row7 col3" >23</td>
                        <td id="T_b9d53_row7_col4" class="data row7 col4" >50</td>
                        <td id="T_b9d53_row7_col5" class="data row7 col5" >10</td>
                        <td id="T_b9d53_row7_col6" class="data row7 col6" >4</td>
                        <td id="T_b9d53_row7_col7" class="data row7 col7" >5702</td>
                        <td id="T_b9d53_row7_col8" class="data row7 col8" >190</td>
                        <td id="T_b9d53_row7_col9" class="data row7 col9" >203</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_b9d53_row8_col0" class="data row8 col0" >18</td>
                        <td id="T_b9d53_row8_col1" class="data row8 col1" >60</td>
                        <td id="T_b9d53_row8_col2" class="data row8 col2" >42</td>
                        <td id="T_b9d53_row8_col3" class="data row8 col3" >101</td>
                        <td id="T_b9d53_row8_col4" class="data row8 col4" >2</td>
                        <td id="T_b9d53_row8_col5" class="data row8 col5" >120</td>
                        <td id="T_b9d53_row8_col6" class="data row8 col6" >32</td>
                        <td id="T_b9d53_row8_col7" class="data row8 col7" >12</td>
                        <td id="T_b9d53_row8_col8" class="data row8 col8" >5425</td>
                        <td id="T_b9d53_row8_col9" class="data row8 col9" >39</td>
            </tr>
            <tr>
                        <th id="T_b9d53_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_b9d53_row9_col0" class="data row9 col0" >22</td>
                        <td id="T_b9d53_row9_col1" class="data row9 col1" >21</td>
                        <td id="T_b9d53_row9_col2" class="data row9 col2" >29</td>
                        <td id="T_b9d53_row9_col3" class="data row9 col3" >62</td>
                        <td id="T_b9d53_row9_col4" class="data row9 col4" >123</td>
                        <td id="T_b9d53_row9_col5" class="data row9 col5" >35</td>
                        <td id="T_b9d53_row9_col6" class="data row9 col6" >1</td>
                        <td id="T_b9d53_row9_col7" class="data row9 col7" >170</td>
                        <td id="T_b9d53_row9_col8" class="data row9 col8" >361</td>
                        <td id="T_b9d53_row9_col9" class="data row9 col9" >5125</td>
            </tr>
    </tbody></table>



Another way of thinking it, we want the values (color) to converge into the diagonal center of each number, whether it be column or row perspective. From the table above, we get to see that column 8 has more of the values spread out horizontally. This tells us that the model is having a hard time predicting the number 8. In other words, it is incorrectly predicting numbers to be number 8 where it is really is not. Looking at the number it from the perspective of its row, because it is being less precise, it is getting almost all the number 8s correctly.


```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
pd.DataFrame(norm_conf_mx * 100).style.bar(vmax=10, vmin=.5)
```




<style  type="text/css" >
#T_c2e77_row0_col0,#T_c2e77_row0_col1,#T_c2e77_row0_col2,#T_c2e77_row0_col3,#T_c2e77_row0_col4,#T_c2e77_row0_col7,#T_c2e77_row0_col9,#T_c2e77_row1_col0,#T_c2e77_row1_col1,#T_c2e77_row1_col3,#T_c2e77_row1_col4,#T_c2e77_row1_col6,#T_c2e77_row1_col7,#T_c2e77_row1_col9,#T_c2e77_row2_col0,#T_c2e77_row2_col2,#T_c2e77_row2_col5,#T_c2e77_row2_col9,#T_c2e77_row3_col0,#T_c2e77_row3_col1,#T_c2e77_row3_col3,#T_c2e77_row3_col4,#T_c2e77_row3_col6,#T_c2e77_row4_col0,#T_c2e77_row4_col1,#T_c2e77_row4_col3,#T_c2e77_row4_col4,#T_c2e77_row4_col5,#T_c2e77_row4_col7,#T_c2e77_row5_col1,#T_c2e77_row5_col5,#T_c2e77_row5_col7,#T_c2e77_row6_col0,#T_c2e77_row6_col1,#T_c2e77_row6_col3,#T_c2e77_row6_col6,#T_c2e77_row6_col7,#T_c2e77_row6_col9,#T_c2e77_row7_col0,#T_c2e77_row7_col1,#T_c2e77_row7_col3,#T_c2e77_row7_col5,#T_c2e77_row7_col6,#T_c2e77_row7_col7,#T_c2e77_row8_col0,#T_c2e77_row8_col4,#T_c2e77_row8_col7,#T_c2e77_row8_col8,#T_c2e77_row9_col0,#T_c2e77_row9_col1,#T_c2e77_row9_col2,#T_c2e77_row9_col6,#T_c2e77_row9_col9{
            width:  10em;
             height:  80%;
        }#T_c2e77_row0_col5,#T_c2e77_row6_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.7%, transparent 1.7%);
        }#T_c2e77_row0_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.6%, transparent 0.6%);
        }#T_c2e77_row0_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 36.3%, transparent 36.3%);
        }#T_c2e77_row1_col2,#T_c2e77_row2_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.6%, transparent 1.6%);
        }#T_c2e77_row1_col5,#T_c2e77_row8_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.8%, transparent 1.8%);
        }#T_c2e77_row1_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 25.7%, transparent 25.7%);
        }#T_c2e77_row2_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.4%, transparent 0.4%);
        }#T_c2e77_row2_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 10.3%, transparent 10.3%);
        }#T_c2e77_row2_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 7.3%, transparent 7.3%);
        }#T_c2e77_row2_col6,#T_c2e77_row9_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 5.7%, transparent 5.7%);
        }#T_c2e77_row2_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 59.4%, transparent 59.4%);
        }#T_c2e77_row3_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 14.5%, transparent 14.5%);
        }#T_c2e77_row3_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.6%, transparent 28.6%);
        }#T_c2e77_row3_col7,#T_c2e77_row8_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 2.3%, transparent 2.3%);
        }#T_c2e77_row3_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 67.7%, transparent 67.7%);
        }#T_c2e77_row3_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 5.6%, transparent 5.6%);
        }#T_c2e77_row4_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.0%, transparent 3.0%);
        }#T_c2e77_row4_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 1.0%, transparent 1.0%);
        }#T_c2e77_row4_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 54.4%, transparent 54.4%);
        }#T_c2e77_row4_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 22.5%, transparent 22.5%);
        }#T_c2e77_row5_col0{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.2%, transparent 0.2%);
        }#T_c2e77_row5_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.8%, transparent 0.8%);
        }#T_c2e77_row5_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 22.1%, transparent 22.1%);
        }#T_c2e77_row5_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 4.8%, transparent 4.8%);
        }#T_c2e77_row5_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 9.9%, transparent 9.9%);
        }#T_c2e77_row5_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 96.3%, transparent 96.3%);
        }#T_c2e77_row5_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 7.4%, transparent 7.4%);
        }#T_c2e77_row6_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.6%, transparent 3.6%);
        }#T_c2e77_row6_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 10.2%, transparent 10.2%);
        }#T_c2e77_row6_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 17.1%, transparent 17.1%);
        }#T_c2e77_row7_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.5%, transparent 3.5%);
        }#T_c2e77_row7_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.1%, transparent 3.1%);
        }#T_c2e77_row7_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 26.7%, transparent 26.7%);
        }#T_c2e77_row7_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.8%, transparent 28.8%);
        }#T_c2e77_row8_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 5.5%, transparent 5.5%);
        }#T_c2e77_row8_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 12.9%, transparent 12.9%);
        }#T_c2e77_row8_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 16.3%, transparent 16.3%);
        }#T_c2e77_row8_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.5%, transparent 0.5%);
        }#T_c2e77_row9_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 16.5%, transparent 16.5%);
        }#T_c2e77_row9_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 0.9%, transparent 0.9%);
        }#T_c2e77_row9_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 24.8%, transparent 24.8%);
        }#T_c2e77_row9_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 58.6%, transparent 58.6%);
        }</style><table id="T_c2e77_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >0</th>        <th class="col_heading level0 col1" >1</th>        <th class="col_heading level0 col2" >2</th>        <th class="col_heading level0 col3" >3</th>        <th class="col_heading level0 col4" >4</th>        <th class="col_heading level0 col5" >5</th>        <th class="col_heading level0 col6" >6</th>        <th class="col_heading level0 col7" >7</th>        <th class="col_heading level0 col8" >8</th>        <th class="col_heading level0 col9" >9</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c2e77_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_c2e77_row0_col0" class="data row0 col0" >0.000000</td>
                        <td id="T_c2e77_row0_col1" class="data row0 col1" >0.000000</td>
                        <td id="T_c2e77_row0_col2" class="data row0 col2" >0.320783</td>
                        <td id="T_c2e77_row0_col3" class="data row0 col3" >0.084417</td>
                        <td id="T_c2e77_row0_col4" class="data row0 col4" >0.168833</td>
                        <td id="T_c2e77_row0_col5" class="data row0 col5" >0.658450</td>
                        <td id="T_c2e77_row0_col6" class="data row0 col6" >0.557150</td>
                        <td id="T_c2e77_row0_col7" class="data row0 col7" >0.084417</td>
                        <td id="T_c2e77_row0_col8" class="data row0 col8" >3.950701</td>
                        <td id="T_c2e77_row0_col9" class="data row0 col9" >0.016883</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_c2e77_row1_col0" class="data row1 col0" >0.014832</td>
                        <td id="T_c2e77_row1_col1" class="data row1 col1" >0.000000</td>
                        <td id="T_c2e77_row1_col2" class="data row1 col2" >0.652625</td>
                        <td id="T_c2e77_row1_col3" class="data row1 col3" >0.252151</td>
                        <td id="T_c2e77_row1_col4" class="data row1 col4" >0.059330</td>
                        <td id="T_c2e77_row1_col5" class="data row1 col5" >0.667458</td>
                        <td id="T_c2e77_row1_col6" class="data row1 col6" >0.059330</td>
                        <td id="T_c2e77_row1_col7" class="data row1 col7" >0.118659</td>
                        <td id="T_c2e77_row1_col8" class="data row1 col8" >2.936814</td>
                        <td id="T_c2e77_row1_col9" class="data row1 col9" >0.118659</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_c2e77_row2_col0" class="data row2 col0" >0.386036</td>
                        <td id="T_c2e77_row2_col1" class="data row2 col1" >0.537093</td>
                        <td id="T_c2e77_row2_col2" class="data row2 col2" >0.000000</td>
                        <td id="T_c2e77_row2_col3" class="data row2 col3" >1.477006</td>
                        <td id="T_c2e77_row2_col4" class="data row2 col4" >1.191675</td>
                        <td id="T_c2e77_row2_col5" class="data row2 col5" >0.335683</td>
                        <td id="T_c2e77_row2_col6" class="data row2 col6" >1.040618</td>
                        <td id="T_c2e77_row2_col7" class="data row2 col7" >0.654582</td>
                        <td id="T_c2e77_row2_col8" class="data row2 col8" >6.143001</td>
                        <td id="T_c2e77_row2_col9" class="data row2 col9" >0.134273</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_c2e77_row3_col0" class="data row3 col0" >0.440385</td>
                        <td id="T_c2e77_row3_col1" class="data row3 col1" >0.326211</td>
                        <td id="T_c2e77_row3_col2" class="data row3 col2" >1.875714</td>
                        <td id="T_c2e77_row3_col3" class="data row3 col3" >0.000000</td>
                        <td id="T_c2e77_row3_col4" class="data row3 col4" >0.000000</td>
                        <td id="T_c2e77_row3_col5" class="data row3 col5" >3.213179</td>
                        <td id="T_c2e77_row3_col6" class="data row3 col6" >0.391453</td>
                        <td id="T_c2e77_row3_col7" class="data row3 col7" >0.717664</td>
                        <td id="T_c2e77_row3_col8" class="data row3 col8" >6.931985</td>
                        <td id="T_c2e77_row3_col9" class="data row3 col9" >1.027565</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_c2e77_row4_col0" class="data row4 col0" >0.119822</td>
                        <td id="T_c2e77_row4_col1" class="data row4 col1" >0.239644</td>
                        <td id="T_c2e77_row4_col2" class="data row4 col2" >0.787402</td>
                        <td id="T_c2e77_row4_col3" class="data row4 col3" >0.188292</td>
                        <td id="T_c2e77_row4_col4" class="data row4 col4" >0.000000</td>
                        <td id="T_c2e77_row4_col5" class="data row4 col5" >0.136939</td>
                        <td id="T_c2e77_row4_col6" class="data row4 col6" >0.599110</td>
                        <td id="T_c2e77_row4_col7" class="data row4 col7" >0.342349</td>
                        <td id="T_c2e77_row4_col8" class="data row4 col8" >5.665868</td>
                        <td id="T_c2e77_row4_col9" class="data row4 col9" >2.636084</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_c2e77_row5_col0" class="data row5 col0" >0.516510</td>
                        <td id="T_c2e77_row5_col1" class="data row5 col1" >0.350489</td>
                        <td id="T_c2e77_row5_col2" class="data row5 col2" >0.571850</td>
                        <td id="T_c2e77_row5_col3" class="data row5 col3" >2.600996</td>
                        <td id="T_c2e77_row5_col4" class="data row5 col4" >0.959233</td>
                        <td id="T_c2e77_row5_col5" class="data row5 col5" >0.000000</td>
                        <td id="T_c2e77_row5_col6" class="data row5 col6" >1.438849</td>
                        <td id="T_c2e77_row5_col7" class="data row5 col7" >0.332042</td>
                        <td id="T_c2e77_row5_col8" class="data row5 col8" >9.647666</td>
                        <td id="T_c2e77_row5_col9" class="data row5 col9" >1.199041</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_c2e77_row6_col0" class="data row6 col0" >0.456235</td>
                        <td id="T_c2e77_row6_col1" class="data row6 col1" >0.270362</td>
                        <td id="T_c2e77_row6_col2" class="data row6 col2" >0.844880</td>
                        <td id="T_c2e77_row6_col3" class="data row6 col3" >0.033795</td>
                        <td id="T_c2e77_row6_col4" class="data row6 col4" >0.659006</td>
                        <td id="T_c2e77_row6_col5" class="data row6 col5" >1.470091</td>
                        <td id="T_c2e77_row6_col6" class="data row6 col6" >0.000000</td>
                        <td id="T_c2e77_row6_col7" class="data row6 col7" >0.118283</td>
                        <td id="T_c2e77_row6_col8" class="data row6 col8" >2.129098</td>
                        <td id="T_c2e77_row6_col9" class="data row6 col9" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_c2e77_row7_col0" class="data row7 col0" >0.319234</td>
                        <td id="T_c2e77_row7_col1" class="data row7 col1" >0.175579</td>
                        <td id="T_c2e77_row7_col2" class="data row7 col2" >0.830008</td>
                        <td id="T_c2e77_row7_col3" class="data row7 col3" >0.367119</td>
                        <td id="T_c2e77_row7_col4" class="data row7 col4" >0.798085</td>
                        <td id="T_c2e77_row7_col5" class="data row7 col5" >0.159617</td>
                        <td id="T_c2e77_row7_col6" class="data row7 col6" >0.063847</td>
                        <td id="T_c2e77_row7_col7" class="data row7 col7" >0.000000</td>
                        <td id="T_c2e77_row7_col8" class="data row7 col8" >3.032721</td>
                        <td id="T_c2e77_row7_col9" class="data row7 col9" >3.240223</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_c2e77_row8_col0" class="data row8 col0" >0.307640</td>
                        <td id="T_c2e77_row8_col1" class="data row8 col1" >1.025466</td>
                        <td id="T_c2e77_row8_col2" class="data row8 col2" >0.717826</td>
                        <td id="T_c2e77_row8_col3" class="data row8 col3" >1.726201</td>
                        <td id="T_c2e77_row8_col4" class="data row8 col4" >0.034182</td>
                        <td id="T_c2e77_row8_col5" class="data row8 col5" >2.050931</td>
                        <td id="T_c2e77_row8_col6" class="data row8 col6" >0.546915</td>
                        <td id="T_c2e77_row8_col7" class="data row8 col7" >0.205093</td>
                        <td id="T_c2e77_row8_col8" class="data row8 col8" >0.000000</td>
                        <td id="T_c2e77_row8_col9" class="data row8 col9" >0.666553</td>
            </tr>
            <tr>
                        <th id="T_c2e77_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_c2e77_row9_col0" class="data row9 col0" >0.369810</td>
                        <td id="T_c2e77_row9_col1" class="data row9 col1" >0.353001</td>
                        <td id="T_c2e77_row9_col2" class="data row9 col2" >0.487477</td>
                        <td id="T_c2e77_row9_col3" class="data row9 col3" >1.042192</td>
                        <td id="T_c2e77_row9_col4" class="data row9 col4" >2.067574</td>
                        <td id="T_c2e77_row9_col5" class="data row9 col5" >0.588334</td>
                        <td id="T_c2e77_row9_col6" class="data row9 col6" >0.016810</td>
                        <td id="T_c2e77_row9_col7" class="data row9 col7" >2.857623</td>
                        <td id="T_c2e77_row9_col8" class="data row9 col8" >6.068247</td>
                        <td id="T_c2e77_row9_col9" class="data row9 col9" >0.000000</td>
            </tr>
    </tbody></table>



Here, we remove the diagonal values and divide each row by its sums. This will give us a better view of where the values are on the matrix. We get to see that the model really is having a hard time making correct predictions of
