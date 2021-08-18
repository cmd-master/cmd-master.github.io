---
layout: post
date: 2021-08-19 3:44
title:  "Predicting Housing Prices"
mood: happy
category:
- python
- machine learning
---

![](https://raw.githubusercontent.com/cmd-master/agl-housing-prices/885ff17fb15a7b12289f9ab98f44821a8a0a698b/hopr-matrix-annot.svg)

Finding a House is never easy. This analysis is about predicting the price of a House in California using Machine Learning.

<!--more-->

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
pd.options.display.max_rows = 1000; pd.options.display.max_columns = 100;
sns.set_style('whitegrid')
import kaggle
from zipfile import ZipFile
```

Configure


```python
train = pd.read_csv(ZipFile("house-prices-advanced-regression-techniques.zip").open("train.csv"))
test = pd.read_csv(ZipFile("house-prices-advanced-regression-techniques.zip").open("test.csv"))
```

Load data


```python
train.head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



Preview head


```python
numeric = [f for f in train.columns if train.dtypes[f] != 'object']
numeric.remove('SalePrice')
numeric.remove('Id')
categorical = [f for f in train.columns if train.dtypes[f] == 'object']
target = 'SalePrice'
```

Separate numeric, categorical and target from each other.

# Explore Data
1460 ids in the training set. 81 variables in the dataset. 36 numeric variables (excluding Ids and Sale Price) and 43 categorical variables.

Numeric:
'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
'2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
'MoSold', 'YrSold'

Categorical:
'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
'MiscFeature', 'SaleType', 'SaleCondition'

## Sale Price Variable
Sale Price is what we want to predict. Mean is around 180k with a couple of outliers.


```python
y = train['SalePrice']
y.hist(bins=100)
plt.savefig('hopr-saleprice.svg')
plt.close();
```

![](hopr-saleprice-ant.svg)
See the shape of the target data.


```python
y.describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64



See a bit of a right skew due to some outliers. Will it affect ML accuracy if we keep them in, not sure...

## Overall Quality
SalePrice increases as OverallQual increase, we know this. Because we are working with averages, a feature with outliers that are really far from the rest of the points will skew the our algorithm.


```python
sns.boxplot(x=train['OverallQual'], y=train['SalePrice'], color='steelblue')
plt.savefig('hopr-overallquality.svg')
plt.close();
```

![](https://raw.githubusercontent.com/cmd-master/agl-housing-prices/885ff17fb15a7b12289f9ab98f44821a8a0a698b/hopr-overallquality-ant.svg)
In this dataset, we found some but aren't far enough or numerous enough to become a threat to our averages.

## GrLivArea
GrLivArea is "Above Grade Living Area". There are many ways to measure a house but the most frequently measurement is to measure the area that is on the ground level. In the dataset, it has a strong relationship with SalePrice.


```python
sns.regplot(x=train['GrLivArea'], y=train['SalePrice'], ci=None
            , scatter_kws={'alpha':0.5}
            , line_kws={"color": "orange"})

plt.savefig('hopr-saleprice-grlivarea.svg')
plt.close();
```

![](https://raw.githubusercontent.com/cmd-master/agl-housing-prices/885ff17fb15a7b12289f9ab98f44821a8a0a698b/hopr-saleprice-grlivarea-ant.svg)
The larger the house, the more you pay. Most of the houses follows this idea except for 2 houses priced at 523 and 1298.

## Correlation
Now we look at what features is more correlated with SalePrice.


```python
g = (train[numeric].corrwith(train['SalePrice']).sort_values(ascending=False)
     .reset_index()).head(15)
g.columns = ['Features', 'Correlation']
g
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
      <th>Features</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>OverallQual</td>
      <td>0.790982</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GrLivArea</td>
      <td>0.708624</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GarageCars</td>
      <td>0.640409</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GarageArea</td>
      <td>0.623431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TotalBsmtSF</td>
      <td>0.613581</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1stFlrSF</td>
      <td>0.605852</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FullBath</td>
      <td>0.560664</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TotRmsAbvGrd</td>
      <td>0.533723</td>
    </tr>
    <tr>
      <th>8</th>
      <td>YearBuilt</td>
      <td>0.522897</td>
    </tr>
    <tr>
      <th>9</th>
      <td>YearRemodAdd</td>
      <td>0.507101</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GarageYrBlt</td>
      <td>0.486362</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MasVnrArea</td>
      <td>0.477493</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Fireplaces</td>
      <td>0.466929</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BsmtFinSF1</td>
      <td>0.386420</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LotFrontage</td>
      <td>0.351799</td>
    </tr>
  </tbody>
</table>
</div>



Overall Quality seems to be the most correlated matric to Sales Price.


```python
matrix = train[['SalePrice'] + list(g['Features'])[:15]].corr()
np.fill_diagonal(matrix.values, 0)
sns.heatmap(matrix, linewidths=.01, linecolor='lightgrey'
            , annot=True, cmap='Blues', vmin=.4, cbar=False);
plt.savefig('hopr-matrix.svg')
plt.close();
```

![](https://raw.githubusercontent.com/cmd-master/agl-housing-prices/885ff17fb15a7b12289f9ab98f44821a8a0a698b/hopr-matrix-annot.svg)

Calculated correlation with the rest of the features.
- GarageCars and GarageArea has a high correlation with each other.
- There are others, which I will keep in mind when I decide to combine them to create new combined features.

## Missing Data
We look at how much data we are missing.


```python
missing = train.isnull().sum()
missing = missing[missing>0]
missing_tally = pd.DataFrame({
    'count': missing,
    'proportion': missing / len(train) * 100
})
missing_tally.sort_values(by='proportion', ascending=False)
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
      <th>count</th>
      <th>proportion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>99.520548</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>96.301370</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>93.767123</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>80.753425</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>47.260274</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>17.739726</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>5.547945</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>5.547945</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>5.547945</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>5.547945</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>5.547945</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>2.602740</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>2.602740</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>2.534247</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>2.534247</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>2.534247</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.547945</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.547945</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.068493</td>
    </tr>
  </tbody>
</table>
</div>



19 features have missing data. Seems like missing data in PoolQC was used to indicate that there are no pools for those houses. Need to fill them up.

## Batch Exploring Features



```python
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    x=plt.xticks(rotation=90)

f = pd.melt(train, id_vars=['SalePrice'], value_vars=categorical)

g = sns.FacetGrid(f.fillna('NULL'), col="variable", col_wrap=2
                  , height=4, sharex=False, sharey=False)
g = g.map(boxplot, "value", "SalePrice")
```



![png](output_29_0.png)



- **Neighborhood Split Class**. CollgCr seems to be average. OldTown and Edwards are commonly cheap. NridgHt, NoRidge and StonBr are at the higher end.
- **Poor-Excellent Quality**. Some categories can be turned into numeric values.
- **SaleCondition and SaleType**. Partial SalesCondition and New SaleType seems to be both at higher value.


# Cleaning Area
This is where I clean the data.


```python
from sklearn.impute import SimpleImputer
imp_categorical = SimpleImputer(strategy='constant', fill_value='NULL') #init
imp_numeric = SimpleImputer(strategy="median") #init
```

Simple Imputer that fills in missing values as null for categorical and median for numeric.


```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
```

DataFrameSelectors that selects particular columns for the Data Pipeline.


```python
from sklearn.preprocessing import OrdinalEncoder
set_order = [np.array(train.fillna('NULL').groupby(feat)[target].mean()
                   .rank().sort_values().keys()) for feat in categorical ]
set_order = [np.append(feat, 'NULL') if 'NULL' not in feat else feat for feat in set_order  ]
enc_order = OrdinalEncoder(categories = set_order
#                            , handle_unknown='use_encoded_value'
#                            , unknown_value=np.nan
                          ) #init
```

Using Ordinal Encoder of sklearn to replace categorical values with the rank values based on the mean salesprice of each.

## Feature Engineering


```python
# train_prep['GarageCars_GarageArea'] = train_prep['GarageCars']/train_prep['GarageArea']

# train_prep['TotRmsAbvGrd_GrLivArea'] = train_prep['GrLivArea'] /train_prep['TotRmsAbvGrd']

# train_prep['TotalBsmtSF_1stFlrSF'] = train_prep['TotalBsmtSF'] /train_prep['1stFlrSF']

# train_prep['YearBuilt_GarageYrBlt'] = train_prep['YearBuilt'] / train_prep['GarageYrBlt']

# train_prep.corrwith(train['SalePrice']).sort_values(ascending=False)

# categorical = categorical + ['TotRmsAbvGrd_GrLivArea','TotalBsmtSF_1stFlrSF', 'YearBuilt_GarageYrBlt' ]
```

I tried do feature engineering but it did not improve the scores. But this is something worth looking at later.

## Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

## Pipeline


```python
pipe_cat = Pipeline([
    ('selector', DataFrameSelector(categorical))
    , ('imputer_cat', imp_categorical)
    , ('enc_setorder', enc_order)
#     , ('scaler', scaler)
])

pipe_num = Pipeline([
    ('selector', DataFrameSelector(numeric))
    , ('imputer_num', imp_numeric)
    , ('scaler', scaler)

])

pipe_full = FeatureUnion([
    ('pipe_cat', pipe_cat),
    ('pipe_num', pipe_num)    
])

train_prep = pd.DataFrame(pipe_full.fit_transform(train[categorical + numeric])
                          , columns = categorical + numeric)
```

Pipeline that transforms data based on the training data.

### Checks


```python
(pd.DataFrame({'label': train['PoolQC'].fillna('NULL'), 'val': train_prep['PoolQC']})
 .groupby('label')['val'].max().sort_values(ascending=False))
```




    label
    Ex      3.0
    Fa      2.0
    Gd      1.0
    NULL    0.0
    Name: val, dtype: float64



# Machine Learning
## 3 Models


```python
features = categorical + numeric
```

### Linear Regression


```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_reg = LinearRegression()
lin_reg.fit(train_prep[features], train[target])
lin_mse = mean_squared_error(train[target], lin_reg.predict(train_prep[features]))
np.sqrt(lin_mse)
```




    32004.827491259395



Best score 29895

### Decision Tree


```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_prep[features], train[target])
tree_mse = mean_squared_error(train[target], tree_reg.predict(train_prep[features]))
np.sqrt(tree_mse)
```




    0.0



Overfitting

### Random Forest


```python
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(train_prep[features], train[target])
forest_mse = mean_squared_error(train[target], forest_reg.predict(train_prep[features]))
np.sqrt(forest_mse)
```




    10664.059267714101



Best Score 10142

## Cross Validation


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, train_prep[features], train[target]
                        , scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print('Mean: ', rmse_scores.mean())
print('STD: ', rmse_scores.std())
```

    [2.87575443e+04 3.27923086e+04 2.69836180e+04 4.01914752e+04
     3.59258971e+04 2.84730800e+04 1.63784652e+16 2.82502775e+04
     6.33038648e+04 3.39853807e+04]
    Mean:  1637846521935064.0
    STD:  4913539565698971.0


Linear Regression does not have a good score.


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, train_prep[features], train[target]
                        , scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print('Mean: ', rmse_scores.mean())
print('STD: ', rmse_scores.std())
```

    [32140.41033401 39028.18501654 37471.28426286 48955.6173641
     39983.73921067 32470.02528023 33490.43864248 38314.10177862
     68916.35902821 35697.81313493]
    Mean:  40646.79740526516
    STD:  10496.254797502597


Decision Tree seems to have a good score.


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, train_prep[features], train[target]
                        , scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
print('Mean: ', rmse_scores.mean())
print('STD: ', rmse_scores.std())
```

    [21511.45784928 25289.56312317 23132.18116241 37086.63171041
     32303.24974653 25879.85435396 22880.22690136 22357.71093001
     38635.85688905 27897.74619761]
    Mean:  27697.447886377486
    STD:  5907.513788755686


Random Forest shows a better score.

## Grid Search


```python
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(train_prep[features], train[target])
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 scoring='neg_mean_squared_error')



We do a grid search to try out certain parameters to tweak our Random Forest model.


```python
grid_search.best_params_
```




    {'max_features': 8, 'n_estimators': 30}



This seems to be the parameters that would give us a decent score.


```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    41395.05189781541 {'max_features': 2, 'n_estimators': 3}
    34116.375427621926 {'max_features': 2, 'n_estimators': 10}
    33058.32499049918 {'max_features': 2, 'n_estimators': 30}
    37349.08382841939 {'max_features': 4, 'n_estimators': 3}
    32445.01067763226 {'max_features': 4, 'n_estimators': 10}
    31158.95839865736 {'max_features': 4, 'n_estimators': 30}
    37379.702835454635 {'max_features': 6, 'n_estimators': 3}
    31190.751111923968 {'max_features': 6, 'n_estimators': 10}
    30431.609549032695 {'max_features': 6, 'n_estimators': 30}
    36688.3723227568 {'max_features': 8, 'n_estimators': 3}
    31267.35332881621 {'max_features': 8, 'n_estimators': 10}
    28611.764424002107 {'max_features': 8, 'n_estimators': 30}
    37258.75260794741 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    33160.52802993125 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    38217.784195715605 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    30551.90254099896 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    36869.17852863255 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    31748.886282288036 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}


29447.33705509524 {'max_features': 8, 'n_estimators': 30}

## Test


```python
final_model = grid_search.best_estimator_
test_prep = pipe_full.transform(test)
test_prep = pd.DataFrame(pipe_full.transform(test[categorical + numeric])
                          , columns = categorical + numeric)
final_predictions = final_model.predict(test_prep.fillna(0))

submission = pd.DataFrame({'Id': test['Id'], 'SalePrice':final_predictions})

submission.to_csv('mysubmission.csv', index=False)
```

Best Score 0.14943 Rank 9532
