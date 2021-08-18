---
layout: post
date: 2021-08-01 5:21
title:  "Machine Learning End to End"
mood: happy
category:
- python
- machine learning
---

![confusion_matrix](https://raw.githubusercontent.com/cmd-master/std-handsOnMachineLearning/7022e99069e03cd91ea5cd08992409b7f00cc6b4/image1.svg)

This is on Chapter 3: Classification, where it explains the fundamentals on how Machine Learning identify one group from the other using statistics.

Notes for *Hands on Machine Learning with Scikit by Aurelien Geron*.

<!--more-->

```python
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
df = pd.read_csv('housing-Copy1.csv')
df.head()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.keys()
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
           'median_house_value', 'ocean_proximity'],
          dtype='object')




```python
num = [f for f in df.columns if df.dtypes[f] != 'object']
cat = [f for f in df.columns if df.dtypes[f] == 'object']
num.remove('median_house_value')
target = 'median_house_value' # Target. What we want to predict
```

# Preview
20640 rows with 10 variables. 8 numeric and 1 categorical. *median_house_value* is what we are trying to predict.
- Numeric: 'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income'
- Categorical: 'ocean_proximity'
- Target: 'median_house_value'


```python
df.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Target


```python
df[target].hist(bins=100);
```



![png](https://raw.githubusercontent.com/cmd-master/std-handson/0abb7c21efd970ff79340486da40fa56811ce871/target-housing.png)




```python
df[target].describe()
```




    count     20640.000000
    mean     206855.816909
    std      115395.615874
    min       14999.000000
    25%      119600.000000
    50%      179700.000000
    75%      264725.000000
    max      500001.000000
    Name: median_house_value, dtype: float64



The **median_house_value** is our target and there is a cap in the target at 500,000.

# Create A Test Set
When splitting data, check first if there are similar groups you want to stratify to get an even random split.


```python
## Group median income according to 5 categories
df['income_cat'] = np.ceil(df[target] / 1.5)
df['income_cat'].where(df['income_cat'] < 5, 5.0, inplace=True)
num = num + ['income_cat']
```


```python
## Using Startified Split
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    train = df.loc[train_index]
    test = df.loc[test_index]
```


```python
df.income_cat.value_counts() / len(df)
```




    5.0    1.0
    Name: income_cat, dtype: float64




```python
train.income_cat.value_counts() / len(train)
```




    5.0    1.0
    Name: income_cat, dtype: float64




```python
test.income_cat.value_counts() / len(test)
```




    5.0    1.0
    Name: income_cat, dtype: float64



The _Test Set_ generated using stratified sampling has income category proportions almost identical to those in the full dataset, whereas the _Test Set_ generated using purely random sampling is quite skewed.

You now have the Train dataset. From this point forward, all cleaning you do must be only done on Train. Test should not be touched because of it lead to overfitting when we come to evaluation.

# Explore Data
Start with checking correlations of numeric variables with the target data. Then make explore the data with many graphs but only show the top 3 graphs. Pairplot, scatterplot and a wildcard graph. Use only the train data in exploring.

## Correlations


```python
train.corr()[target].sort_values(ascending=False).reset_index().style.bar()
```




<style  type="text/css" >
#T_02fcd_row0_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 100.0%, transparent 100.0%);
        }#T_02fcd_row1_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 73.0%, transparent 73.0%);
        }#T_02fcd_row2_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 24.8%, transparent 24.8%);
        }#T_02fcd_row3_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 22.2%, transparent 22.2%);
        }#T_02fcd_row4_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 18.9%, transparent 18.9%);
        }#T_02fcd_row5_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 17.5%, transparent 17.5%);
        }#T_02fcd_row6_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 10.8%, transparent 10.8%);
        }#T_02fcd_row7_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 9.3%, transparent 9.3%);
        }#T_02fcd_row8_col1{
            width:  10em;
             height:  80%;
        }</style><table id="T_02fcd_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >index</th>        <th class="col_heading level0 col1" >median_house_value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_02fcd_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_02fcd_row0_col0" class="data row0 col0" >median_house_value</td>
                        <td id="T_02fcd_row0_col1" class="data row0 col1" >1.000000</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_02fcd_row1_col0" class="data row1 col0" >median_income</td>
                        <td id="T_02fcd_row1_col1" class="data row1 col1" >0.689752</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_02fcd_row2_col0" class="data row2 col0" >total_rooms</td>
                        <td id="T_02fcd_row2_col1" class="data row2 col1" >0.136167</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_02fcd_row3_col0" class="data row3 col0" >housing_median_age</td>
                        <td id="T_02fcd_row3_col1" class="data row3 col1" >0.107050</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_02fcd_row4_col0" class="data row4 col0" >households</td>
                        <td id="T_02fcd_row4_col1" class="data row4 col1" >0.068976</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_02fcd_row5_col0" class="data row5 col0" >total_bedrooms</td>
                        <td id="T_02fcd_row5_col1" class="data row5 col1" >0.052690</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_02fcd_row6_col0" class="data row6 col0" >population</td>
                        <td id="T_02fcd_row6_col1" class="data row6 col1" >-0.024618</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_02fcd_row7_col0" class="data row7 col0" >longitude</td>
                        <td id="T_02fcd_row7_col1" class="data row7 col1" >-0.041830</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_02fcd_row8_col0" class="data row8 col0" >latitude</td>
                        <td id="T_02fcd_row8_col1" class="data row8 col1" >-0.148438</td>
            </tr>
            <tr>
                        <th id="T_02fcd_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_02fcd_row9_col0" class="data row9 col0" >income_cat</td>
                        <td id="T_02fcd_row9_col1" class="data row9 col1" >nan</td>
            </tr>
    </tbody></table>




```python
train.corr().sort_values(by = target, ascending=False).reset_index().style.bar()
```

    D:\Anaconda\lib\site-packages\pandas\io\formats\style.py:1347: RuntimeWarning: All-NaN slice encountered
      smin = np.nanmin(s.to_numpy()) if vmin is None else vmin
    D:\Anaconda\lib\site-packages\pandas\io\formats\style.py:1348: RuntimeWarning: All-NaN slice encountered
      smax = np.nanmax(s.to_numpy()) if vmax is None else vmax





<style  type="text/css" >
#T_85785_row0_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 45.9%, transparent 45.9%);
        }#T_85785_row0_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 40.3%, transparent 40.3%);
        }#T_85785_row0_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 34.3%, transparent 34.3%);
        }#T_85785_row0_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 36.4%, transparent 36.4%);
        }#T_85785_row0_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.1%, transparent 28.1%);
        }#T_85785_row0_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 21.2%, transparent 21.2%);
        }#T_85785_row0_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.4%, transparent 28.4%);
        }#T_85785_row0_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 72.3%, transparent 72.3%);
        }#T_85785_row0_col9,#T_85785_row1_col8,#T_85785_row2_col4,#T_85785_row3_col3,#T_85785_row4_col7,#T_85785_row5_col5,#T_85785_row6_col6,#T_85785_row7_col1,#T_85785_row8_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 100.0%, transparent 100.0%);
        }#T_85785_row1_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 47.5%, transparent 47.5%);
        }#T_85785_row1_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 43.7%, transparent 43.7%);
        }#T_85785_row1_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 17.6%, transparent 17.6%);
        }#T_85785_row1_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 41.1%, transparent 41.1%);
        }#T_85785_row1_col5,#T_85785_row1_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 23.8%, transparent 23.8%);
        }#T_85785_row1_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 24.4%, transparent 24.4%);
        }#T_85785_row1_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 73.0%, transparent 73.0%);
        }#T_85785_row2_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 50.5%, transparent 50.5%);
        }#T_85785_row2_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 46.0%, transparent 46.0%);
        }#T_85785_row2_col3,#T_85785_row3_col4,#T_85785_row3_col5,#T_85785_row3_col6,#T_85785_row3_col7,#T_85785_row3_col8,#T_85785_row7_col2,#T_85785_row8_col1,#T_85785_row8_col9{
            width:  10em;
             height:  80%;
        }#T_85785_row2_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 94.8%, transparent 94.8%);
        }#T_85785_row2_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 89.4%, transparent 89.4%);
        }#T_85785_row2_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 93.8%, transparent 93.8%);
        }#T_85785_row2_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 28.5%, transparent 28.5%);
        }#T_85785_row2_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 24.8%, transparent 24.8%);
        }#T_85785_row3_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 42.2%, transparent 42.2%);
        }#T_85785_row3_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 48.9%, transparent 48.9%);
        }#T_85785_row3_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 22.2%, transparent 22.2%);
        }#T_85785_row4_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 51.0%, transparent 51.0%);
        }#T_85785_row4_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 44.3%, transparent 44.3%);
        }#T_85785_row4_col3,#T_85785_row6_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 4.3%, transparent 4.3%);
        }#T_85785_row4_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 94.1%, transparent 94.1%);
        }#T_85785_row4_col5,#T_85785_row5_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 98.4%, transparent 98.4%);
        }#T_85785_row4_col6,#T_85785_row6_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 93.2%, transparent 93.2%);
        }#T_85785_row4_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 12.2%, transparent 12.2%);
        }#T_85785_row4_col9,#T_85785_row8_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 18.9%, transparent 18.9%);
        }#T_85785_row5_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 51.7%, transparent 51.7%);
        }#T_85785_row5_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 44.5%, transparent 44.5%);
        }#T_85785_row5_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.0%, transparent 3.0%);
        }#T_85785_row5_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 95.0%, transparent 95.0%);
        }#T_85785_row5_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 90.9%, transparent 90.9%);
        }#T_85785_row5_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 10.3%, transparent 10.3%);
        }#T_85785_row5_col9,#T_85785_row8_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 17.5%, transparent 17.5%);
        }#T_85785_row6_col1{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 53.4%, transparent 53.4%);
        }#T_85785_row6_col2{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 42.3%, transparent 42.3%);
        }#T_85785_row6_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 89.8%, transparent 89.8%);
        }#T_85785_row6_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 91.0%, transparent 91.0%);
        }#T_85785_row6_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 11.4%, transparent 11.4%);
        }#T_85785_row6_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 10.8%, transparent 10.8%);
        }#T_85785_row7_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 18.1%, transparent 18.1%);
        }#T_85785_row7_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 29.9%, transparent 29.9%);
        }#T_85785_row7_col5{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 29.5%, transparent 29.5%);
        }#T_85785_row7_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 30.9%, transparent 30.9%);
        }#T_85785_row7_col7{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 27.5%, transparent 27.5%);
        }#T_85785_row7_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 9.8%, transparent 9.8%);
        }#T_85785_row7_col9{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 9.3%, transparent 9.3%);
        }#T_85785_row8_col3{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 27.6%, transparent 27.6%);
        }#T_85785_row8_col4{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 23.5%, transparent 23.5%);
        }#T_85785_row8_col6{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 14.5%, transparent 14.5%);
        }#T_85785_row8_col8{
            width:  10em;
             height:  80%;
            background:  linear-gradient(90deg,#d65f5f 3.1%, transparent 3.1%);
        }</style><table id="T_85785_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >index</th>        <th class="col_heading level0 col1" >longitude</th>        <th class="col_heading level0 col2" >latitude</th>        <th class="col_heading level0 col3" >housing_median_age</th>        <th class="col_heading level0 col4" >total_rooms</th>        <th class="col_heading level0 col5" >total_bedrooms</th>        <th class="col_heading level0 col6" >population</th>        <th class="col_heading level0 col7" >households</th>        <th class="col_heading level0 col8" >median_income</th>        <th class="col_heading level0 col9" >median_house_value</th>        <th class="col_heading level0 col10" >income_cat</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_85785_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_85785_row0_col0" class="data row0 col0" >median_house_value</td>
                        <td id="T_85785_row0_col1" class="data row0 col1" >-0.041830</td>
                        <td id="T_85785_row0_col2" class="data row0 col2" >-0.148438</td>
                        <td id="T_85785_row0_col3" class="data row0 col3" >0.107050</td>
                        <td id="T_85785_row0_col4" class="data row0 col4" >0.136167</td>
                        <td id="T_85785_row0_col5" class="data row0 col5" >0.052690</td>
                        <td id="T_85785_row0_col6" class="data row0 col6" >-0.024618</td>
                        <td id="T_85785_row0_col7" class="data row0 col7" >0.068976</td>
                        <td id="T_85785_row0_col8" class="data row0 col8" >0.689752</td>
                        <td id="T_85785_row0_col9" class="data row0 col9" >1.000000</td>
                        <td id="T_85785_row0_col10" class="data row0 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_85785_row1_col0" class="data row1 col0" >median_income</td>
                        <td id="T_85785_row1_col1" class="data row1 col1" >-0.009710</td>
                        <td id="T_85785_row1_col2" class="data row1 col2" >-0.084431</td>
                        <td id="T_85785_row1_col3" class="data row1 col3" >-0.119018</td>
                        <td id="T_85785_row1_col4" class="data row1 col4" >0.199796</td>
                        <td id="T_85785_row1_col5" class="data row1 col5" >-0.003400</td>
                        <td id="T_85785_row1_col6" class="data row1 col6" >0.009006</td>
                        <td id="T_85785_row1_col7" class="data row1 col7" >0.016974</td>
                        <td id="T_85785_row1_col8" class="data row1 col8" >1.000000</td>
                        <td id="T_85785_row1_col9" class="data row1 col9" >0.689752</td>
                        <td id="T_85785_row1_col10" class="data row1 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_85785_row2_col0" class="data row2 col0" >total_rooms</td>
                        <td id="T_85785_row2_col1" class="data row2 col1" >0.047739</td>
                        <td id="T_85785_row2_col2" class="data row2 col2" >-0.039526</td>
                        <td id="T_85785_row2_col3" class="data row2 col3" >-0.358473</td>
                        <td id="T_85785_row2_col4" class="data row2 col4" >1.000000</td>
                        <td id="T_85785_row2_col5" class="data row2 col5" >0.931636</td>
                        <td id="T_85785_row2_col6" class="data row2 col6" >0.861698</td>
                        <td id="T_85785_row2_col7" class="data row2 col7" >0.919344</td>
                        <td id="T_85785_row2_col8" class="data row2 col8" >0.199796</td>
                        <td id="T_85785_row2_col9" class="data row2 col9" >0.136167</td>
                        <td id="T_85785_row2_col10" class="data row2 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_85785_row3_col0" class="data row3 col0" >housing_median_age</td>
                        <td id="T_85785_row3_col1" class="data row3 col1" >-0.111981</td>
                        <td id="T_85785_row3_col2" class="data row3 col2" >0.016236</td>
                        <td id="T_85785_row3_col3" class="data row3 col3" >1.000000</td>
                        <td id="T_85785_row3_col4" class="data row3 col4" >-0.358473</td>
                        <td id="T_85785_row3_col5" class="data row3 col5" >-0.317529</td>
                        <td id="T_85785_row3_col6" class="data row3 col6" >-0.299773</td>
                        <td id="T_85785_row3_col7" class="data row3 col7" >-0.300377</td>
                        <td id="T_85785_row3_col8" class="data row3 col8" >-0.119018</td>
                        <td id="T_85785_row3_col9" class="data row3 col9" >0.107050</td>
                        <td id="T_85785_row3_col10" class="data row3 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_85785_row4_col0" class="data row4 col0" >households</td>
                        <td id="T_85785_row4_col1" class="data row4 col1" >0.056796</td>
                        <td id="T_85785_row4_col2" class="data row4 col2" >-0.072787</td>
                        <td id="T_85785_row4_col3" class="data row4 col3" >-0.300377</td>
                        <td id="T_85785_row4_col4" class="data row4 col4" >0.919344</td>
                        <td id="T_85785_row4_col5" class="data row4 col5" >0.979495</td>
                        <td id="T_85785_row4_col6" class="data row4 col6" >0.911096</td>
                        <td id="T_85785_row4_col7" class="data row4 col7" >1.000000</td>
                        <td id="T_85785_row4_col8" class="data row4 col8" >0.016974</td>
                        <td id="T_85785_row4_col9" class="data row4 col9" >0.068976</td>
                        <td id="T_85785_row4_col10" class="data row4 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_85785_row5_col0" class="data row5 col0" >total_bedrooms</td>
                        <td id="T_85785_row5_col1" class="data row5 col1" >0.071221</td>
                        <td id="T_85785_row5_col2" class="data row5 col2" >-0.068902</td>
                        <td id="T_85785_row5_col3" class="data row5 col3" >-0.317529</td>
                        <td id="T_85785_row5_col4" class="data row5 col4" >0.931636</td>
                        <td id="T_85785_row5_col5" class="data row5 col5" >1.000000</td>
                        <td id="T_85785_row5_col6" class="data row5 col6" >0.881112</td>
                        <td id="T_85785_row5_col7" class="data row5 col7" >0.979495</td>
                        <td id="T_85785_row5_col8" class="data row5 col8" >-0.003400</td>
                        <td id="T_85785_row5_col9" class="data row5 col9" >0.052690</td>
                        <td id="T_85785_row5_col10" class="data row5 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_85785_row6_col0" class="data row6 col0" >population</td>
                        <td id="T_85785_row6_col1" class="data row6 col1" >0.102285</td>
                        <td id="T_85785_row6_col2" class="data row6 col2" >-0.111175</td>
                        <td id="T_85785_row6_col3" class="data row6 col3" >-0.299773</td>
                        <td id="T_85785_row6_col4" class="data row6 col4" >0.861698</td>
                        <td id="T_85785_row6_col5" class="data row6 col5" >0.881112</td>
                        <td id="T_85785_row6_col6" class="data row6 col6" >1.000000</td>
                        <td id="T_85785_row6_col7" class="data row6 col7" >0.911096</td>
                        <td id="T_85785_row6_col8" class="data row6 col8" >0.009006</td>
                        <td id="T_85785_row6_col9" class="data row6 col9" >-0.024618</td>
                        <td id="T_85785_row6_col10" class="data row6 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_85785_row7_col0" class="data row7 col0" >longitude</td>
                        <td id="T_85785_row7_col1" class="data row7 col1" >1.000000</td>
                        <td id="T_85785_row7_col2" class="data row7 col2" >-0.924861</td>
                        <td id="T_85785_row7_col3" class="data row7 col3" >-0.111981</td>
                        <td id="T_85785_row7_col4" class="data row7 col4" >0.047739</td>
                        <td id="T_85785_row7_col5" class="data row7 col5" >0.071221</td>
                        <td id="T_85785_row7_col6" class="data row7 col6" >0.102285</td>
                        <td id="T_85785_row7_col7" class="data row7 col7" >0.056796</td>
                        <td id="T_85785_row7_col8" class="data row7 col8" >-0.009710</td>
                        <td id="T_85785_row7_col9" class="data row7 col9" >-0.041830</td>
                        <td id="T_85785_row7_col10" class="data row7 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_85785_row8_col0" class="data row8 col0" >latitude</td>
                        <td id="T_85785_row8_col1" class="data row8 col1" >-0.924861</td>
                        <td id="T_85785_row8_col2" class="data row8 col2" >1.000000</td>
                        <td id="T_85785_row8_col3" class="data row8 col3" >0.016236</td>
                        <td id="T_85785_row8_col4" class="data row8 col4" >-0.039526</td>
                        <td id="T_85785_row8_col5" class="data row8 col5" >-0.068902</td>
                        <td id="T_85785_row8_col6" class="data row8 col6" >-0.111175</td>
                        <td id="T_85785_row8_col7" class="data row8 col7" >-0.072787</td>
                        <td id="T_85785_row8_col8" class="data row8 col8" >-0.084431</td>
                        <td id="T_85785_row8_col9" class="data row8 col9" >-0.148438</td>
                        <td id="T_85785_row8_col10" class="data row8 col10" >nan</td>
            </tr>
            <tr>
                        <th id="T_85785_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_85785_row9_col0" class="data row9 col0" >income_cat</td>
                        <td id="T_85785_row9_col1" class="data row9 col1" >nan</td>
                        <td id="T_85785_row9_col2" class="data row9 col2" >nan</td>
                        <td id="T_85785_row9_col3" class="data row9 col3" >nan</td>
                        <td id="T_85785_row9_col4" class="data row9 col4" >nan</td>
                        <td id="T_85785_row9_col5" class="data row9 col5" >nan</td>
                        <td id="T_85785_row9_col6" class="data row9 col6" >nan</td>
                        <td id="T_85785_row9_col7" class="data row9 col7" >nan</td>
                        <td id="T_85785_row9_col8" class="data row9 col8" >nan</td>
                        <td id="T_85785_row9_col9" class="data row9 col9" >nan</td>
                        <td id="T_85785_row9_col10" class="data row9 col10" >nan</td>
            </tr>
    </tbody></table>



## 3 Graphs


```python
sns.pairplot(train[[target] + num[2:-1]]);
```



![png](output_23_0.png)



Pairplot shows that there are features that correlates with other features. We can possibly combine them to have a much stronger feature that correlates with our target.


```python
sns.regplot(x='median_income', y=target, data=train
            , scatter_kws={'alpha':0.5} , line_kws={"color": "orange"});
```



![png](output_25_0.png)



median income is the most correlated feature for our target. There are lines at 500k where we see the cap but there are also lines between 450k and 350k, which needs to be address before we fit our model.


```python
## Using Seaborn
# sns.relplot(x='longitude', y='latitude', data=train, hue='median_house_value'
#             , palette=plt.get_cmap('jet')
#             , size=train['population']/100, sizes=(50,500), alpha=0.3 );
## Using MatplotLib
train.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=train['population']/100, label='population', c=target,
             cmap=plt.get_cmap('jet'), colorbar=True);
```



![png](output_27_0.png)



Housing prices are very much related to the location and the population density. Clustering algorithm to detect main clusters and add new features that measure the proximity to the cluster centers.

# Data Cleaning
After doing an initial exploration, you must clean the dataset but start doing it only on **train**. As you clean it, you will build a pipeline of functions that you will use to clean the test set and any future dataset.

## Missing Data
Assess each numerical values for missing data and use the imputer to replace them with the median value.


```python
mis = train.isnull().sum()
mis = mis[mis>0]
pd.DataFrame({
    'count': mis,
    'proportion': mis / len(train)
}).sort_values(by='proportion', ascending=False).style.bar()
```




<style  type="text/css" >
#T_905af_row0_col0,#T_905af_row0_col1{
            width:  10em;
             height:  80%;
        }</style><table id="T_905af_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >count</th>        <th class="col_heading level0 col1" >proportion</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_905af_level0_row0" class="row_heading level0 row0" >total_bedrooms</th>
                        <td id="T_905af_row0_col0" class="data row0 col0" >207</td>
                        <td id="T_905af_row0_col1" class="data row0 col1" >0.012536</td>
            </tr>
    </tbody></table>




```python
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(strategy='median')
imp_median.fit(train[num])

pd.DataFrame(imp_median.transform(train[num]), columns=num).head() ## preview. can delete
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-118.01</td>
      <td>33.79</td>
      <td>23.0</td>
      <td>2663.0</td>
      <td>430.0</td>
      <td>1499.0</td>
      <td>403.0</td>
      <td>5.7837</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-117.84</td>
      <td>33.73</td>
      <td>20.0</td>
      <td>2572.0</td>
      <td>732.0</td>
      <td>1534.0</td>
      <td>669.0</td>
      <td>2.4211</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-117.60</td>
      <td>33.87</td>
      <td>15.0</td>
      <td>7626.0</td>
      <td>1570.0</td>
      <td>3823.0</td>
      <td>1415.0</td>
      <td>3.4419</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-117.64</td>
      <td>34.09</td>
      <td>34.0</td>
      <td>2839.0</td>
      <td>659.0</td>
      <td>1822.0</td>
      <td>631.0</td>
      <td>3.0500</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.18</td>
      <td>37.77</td>
      <td>51.0</td>
      <td>2107.0</td>
      <td>471.0</td>
      <td>1173.0</td>
      <td>438.0</td>
      <td>3.2552</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



Seems to be a lot of writing code for something simple. But here, we are getting used to *initiate, fit, transform* for sklearn, which will pay off in the long run once we build our automated pipeline.

## Categorical Encoding
For Categorical values, we need to encode them to zeros and ones.


```python
# train = train.join(pd.get_dummies(train['ocean_proximity']))

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit_transform(train[cat])
```




    array([[1, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           ...,
           [0, 0, 0, 0, 1],
           [0, 0, 0, 1, 0],
           [1, 0, 0, 0, 0]])



When encoding, it seems strange that we don't care of the columne names and just have the values zeros and ones. This must have a purpose later on. Better keep moving.

## Feature Engineering
There will be some features that would have a weak relationship with the target but are strongly correlated with other weak features. Exploring the combination of these features by combining them together may result into a stronger feature that we can use to improve our predictive capability of the target.


```python
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
```


```python
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False) # initiate
housing_extra_attributes = attr_adder.transform(train.values) # transform
```

The more you automate these data preparation steps, the more combinations you can automatically try out, making it much more lifely that you will find a great combination.

## Feature Scaling
Total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15.


### MinMaxScaler Normalization
Transforms the numeric data between 0 and 1.


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # initiate
scaler.fit(train[num]) # fit
minmax = scaler.transform(train[num]) # transform
pd.DataFrame(minmax, columns=num).describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16305.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.476581</td>
      <td>0.327853</td>
      <td>0.542308</td>
      <td>0.066839</td>
      <td>0.083031</td>
      <td>0.049609</td>
      <td>0.081745</td>
      <td>0.232449</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.199256</td>
      <td>0.226671</td>
      <td>0.246869</td>
      <td>0.055535</td>
      <td>0.065198</td>
      <td>0.038838</td>
      <td>0.062566</td>
      <td>0.130775</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.254980</td>
      <td>0.147715</td>
      <td>0.333333</td>
      <td>0.036701</td>
      <td>0.045934</td>
      <td>0.027415</td>
      <td>0.046045</td>
      <td>0.142531</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.583665</td>
      <td>0.182253</td>
      <td>0.549020</td>
      <td>0.053945</td>
      <td>0.067039</td>
      <td>0.040615</td>
      <td>0.067094</td>
      <td>0.209304</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.631474</td>
      <td>0.549416</td>
      <td>0.705882</td>
      <td>0.079963</td>
      <td>0.099783</td>
      <td>0.060117</td>
      <td>0.098874</td>
      <td>0.292641</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Standardization
Uses standard deviation with mean as zero.


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train[num])
pd.DataFrame(scaler.transform(train[num]), columns=num).describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.651200e+04</td>
      <td>1.651200e+04</td>
      <td>1.651200e+04</td>
      <td>1.651200e+04</td>
      <td>1.630500e+04</td>
      <td>1.651200e+04</td>
      <td>1.651200e+04</td>
      <td>1.651200e+04</td>
      <td>16512.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.538505e-15</td>
      <td>7.477868e-16</td>
      <td>3.805634e-17</td>
      <td>-6.044616e-17</td>
      <td>1.101984e-16</td>
      <td>-1.053004e-16</td>
      <td>2.669995e-17</td>
      <td>-1.972744e-16</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000030e+00</td>
      <td>1.000030e+00</td>
      <td>1.000030e+00</td>
      <td>1.000030e+00</td>
      <td>1.000031e+00</td>
      <td>1.000030e+00</td>
      <td>1.000030e+00</td>
      <td>1.000030e+00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.391881e+00</td>
      <td>-1.446426e+00</td>
      <td>-2.196807e+00</td>
      <td>-1.203579e+00</td>
      <td>-1.273565e+00</td>
      <td>-1.277386e+00</td>
      <td>-1.306592e+00</td>
      <td>-1.777519e+00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.112179e+00</td>
      <td>-7.947334e-01</td>
      <td>-8.465243e-01</td>
      <td>-5.427055e-01</td>
      <td>-5.690089e-01</td>
      <td>-5.714807e-01</td>
      <td>-5.706202e-01</td>
      <td>-6.875957e-01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.374377e-01</td>
      <td>-6.423593e-01</td>
      <td>2.718806e-02</td>
      <td>-2.321909e-01</td>
      <td>-2.452939e-01</td>
      <td>-2.316002e-01</td>
      <td>-2.341759e-01</td>
      <td>-1.769856e-01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.773818e-01</td>
      <td>9.774942e-01</td>
      <td>6.626152e-01</td>
      <td>2.363289e-01</td>
      <td>2.569403e-01</td>
      <td>2.705575e-01</td>
      <td>2.737761e-01</td>
      <td>4.602849e-01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.626952e+00</td>
      <td>2.965390e+00</td>
      <td>1.854041e+00</td>
      <td>1.680352e+01</td>
      <td>1.406481e+01</td>
      <td>2.447149e+01</td>
      <td>1.467714e+01</td>
      <td>5.869411e+00</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



You must only *fit* scalers to the **train** data. You must fit it to the whole data set or to the test set because this will cause overfitting when we do evaluation.

# Transformation Pipeline
Ok. This is where you get everything together. Do everything step by step.


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
```


```python
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class NewLabelBinarizer(LabelBinarizer):
    def fit(self, X, y=None):
        return super(NewLabelBinarizer, self).fit(X)
    def transform(self, X, y=None):
        return super(NewLabelBinarizer, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(NewLabelBinarizer, self).fit(X).transform(X)
```


```python
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num))
    , ('imputer', SimpleImputer(strategy='median'))
    , ('attribs_adder', CombinedAttributesAdder())
    , ('std_scaler', StandardScaler())
])
```


```python
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat))
    , ('label_binarizer', NewLabelBinarizer())
])
```


```python
full_pipeline = FeatureUnion([
        ('num_pipeline', num_pipeline)
        , ('cat_pipeline', cat_pipeline)
])
```


```python
housing_prepared = full_pipeline.fit_transform(train)
```

# Machine Learning

## Linear Regression


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
housing_prepared = full_pipeline.transform(train) # does not have target
lin_reg = LinearRegression() # init
lin_reg.fit(housing_prepared, train[target]) # fit
lin_mse = mean_squared_error(train[target], lin_reg.predict(housing_prepared)) # predict evaluate
np.sqrt(lin_mse)
```




    67682.88191615137



This is not ideal

## DecisionTree


```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor() # init
tree_reg.fit(housing_prepared, train[target]) # fit
tree_mse = mean_squared_error(train[target], tree_reg.predict(housing_prepared))# predict evaluate
np.sqrt(tree_mse)
```




    0.0



This is overfitting

## Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor() # init
forest_reg.fit(housing_prepared, train[target]) # fit
forest_mse = mean_squared_error(train[target], forest_reg.predict(housing_prepared))# predict evaluate
np.sqrt(forest_mse)
```




    18566.04007891868



Better model to use.

## Cross Validation


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, train[target],
                        scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print(rmse_scores.mean())
print(rmse_scores.std())
```

    [70434.7081069  72251.00352891 66943.27378691 73474.94589848
     72280.04382175 72612.9464193  69592.63132772 66341.5025707
     72766.29991416 69492.40438281]
    70618.97597576358
    2373.3986095685927



```python
scores = cross_val_score(lin_reg, housing_prepared, train[target],
                        scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print(rmse_scores.mean())
print(rmse_scores.std())
```

    [67609.58112566 68923.20005577 65767.71443054 67107.22038658
     68147.70871523 67422.36125738 68453.48347856 68142.56585259
     69394.88530646 67993.11871622]
    67896.18393249981
    959.3909955206789



```python
scores = cross_val_score(forest_reg, housing_prepared, train[target],
                        scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print(rmse_scores.mean())
print(rmse_scores.std())
```

    [50054.64546488 51966.98284739 45919.14523001 51711.26380141
     49151.6703164  50227.25376101 50329.83655027 48249.90369304
     51100.68438006 49328.87698694]
    49804.02630314154
    1693.3575866495264


# Grid Search
- 3 n_estimators x 4 max_features = 12
- 2 n_estimators x 3 max_features = 6
- 5 folds x 18 combinations = 90 rounds of training


```python
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, train[target])
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 scoring='neg_mean_squared_error')




```python
grid_search.best_params_
```




    {'max_features': 8, 'n_estimators': 30}



This shows the best parameter that made the best results


```python
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=8, n_estimators=30)



You can see the best estimator to be used. Here we only placed 1.


```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    62646.056361329305 {'max_features': 2, 'n_estimators': 3}
    54955.674540654785 {'max_features': 2, 'n_estimators': 10}
    52566.82111651963 {'max_features': 2, 'n_estimators': 30}
    60245.08171510005 {'max_features': 4, 'n_estimators': 3}
    52758.8708791719 {'max_features': 4, 'n_estimators': 10}
    50138.68788619675 {'max_features': 4, 'n_estimators': 30}
    58787.90187597542 {'max_features': 6, 'n_estimators': 3}
    51784.44017781117 {'max_features': 6, 'n_estimators': 10}
    49878.96260437533 {'max_features': 6, 'n_estimators': 30}
    57574.58235216306 {'max_features': 8, 'n_estimators': 3}
    52361.49588431292 {'max_features': 8, 'n_estimators': 10}
    49784.921012711886 {'max_features': 8, 'n_estimators': 30}
    62331.827065439735 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54180.991460193094 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59191.20932429057 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52033.387537663475 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    58822.82814676727 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51716.94169366502 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


You can see all the parameters and their corresponding scores here.


```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([7.17384677e-02, 6.91268559e-02, 4.24156531e-02, 1.63615701e-02,
           1.45932572e-02, 1.53044281e-02, 1.40238998e-02, 3.95354204e-01,
           0.00000000e+00, 4.68102256e-02, 1.10050940e-01, 4.22794957e-02,
           5.64750866e-03, 1.49575120e-01, 8.66826374e-05, 2.81408113e-03,
           3.81761083e-03])




```python
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_one_hot_attribs = list(encoder.classes_)
attributes = num + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.3953542037390721, 'median_income'),
     (0.14957512006638152, 'INLAND'),
     (0.11005093979218608, 'pop_per_hhold'),
     (0.0717384677091464, 'longitude'),
     (0.06912685591025496, 'latitude'),
     (0.04681022560439827, 'rooms_per_hhold'),
     (0.04241565307508381, 'housing_median_age'),
     (0.04227949565842384, 'bedrooms_per_room'),
     (0.01636157010936986, 'total_rooms'),
     (0.01530442812926547, 'population'),
     (0.014593257159429216, 'total_bedrooms'),
     (0.01402389979243357, 'households'),
     (0.0056475086553798275, '<1H OCEAN'),
     (0.0038176108273236843, 'NEAR OCEAN'),
     (0.00281408113444609, 'NEAR BAY'),
     (8.668263740521922e-05, 'ISLAND'),
     (0.0, 'income_cat')]



You can see which features are more stronger. Consider removing some weak ones.

# Evaluate in Test Set


```python
final_model = grid_search.best_estimator_
X_test = test.drop(target, axis=1)
Y_test = test[target]
X_test_prep = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prep)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
```




    49594.522233583404




```python

```
