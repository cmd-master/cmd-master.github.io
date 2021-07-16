---
layout: post
date: 2017-05-29 8:14
title:  "How to Survive Titanic"
mood: happy
category:
- udacity
- R
---
![png](https://raw.githubusercontent.com/cmd-master/dand-titanic-survivors/master/img/output_24_0.png)

On 15 April 1912, Titanic, the largest ship of its time, sank after hitting an iceberg in the North Atlantic Ocean. Of the 2,224 people estimated on board, only 705 survived. Although limited, there were enough lifeboats to save 1,178 people and yet fewer made it.

<!--more-->


## Questions
How likely would a passenger survive the tragedy?

- If you are rich, would you most likely be prioritized?
- "Women and Children First". Does your age or gender influence your chances of survivability?

### Objectives
This study analyzes the likelihood of survivability of passengers on board of the Titanic. The analysis is divided according to Demographics and Social Economic Status. The former will be based on Gender and Age and the latter will be based on Ticket Class and Fare.

### Variables
Dependent Variable: If the passenger survived or not. <br>
Independent Variables: 1. Gender 2. Age 3. Ticket Class 4. Fare. <br>
Null Hypothesis: The likelihood of surviving the event are not influenced by demographics and socio economic status. <br>
Hypothesis: The likelihood of survival is influenced by the demographics and socio economic status of the passengers.

## Data Wrangling
### Data Acquisition
The data provided is a list of names of 891 of the 2,224 passengers with the corresponding information for each on board. Below is the Data Dictionary of the data set from [Kaggle](https://www.kaggle.com/c/titanic/data).

- survival: Survival (0 = No, 1 = Yes)
- pclass: Ticket class (1st = Upper, 2nd = Middle, 3rd = Lower)
- sex: Sex
- Age: Age in years (Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5)
- sibsp: # of siblings / spouse aboard the Titanic (Sibling = brother, sister, stepbrother, stepsister, Spouse = husband, wife (mistresses and fiancés were ignored))
- parch: # of parents / children aboard the Titanic (Parent = mother, father, Child = daughter, son, stepdaughter, stepson, Some children travelled only with a nanny, therefore parch=0 for them.)
- ticket: Ticket number
- fare: Passenger fare
- cabin: Cabin number
- embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)


```python
import pandas as pd
import numpy as np
# Graphing
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
titanic_df = pd.read_csv('titanic-data.csv') # Read CSV and stores in to titanic_df variable.
```

### Data Cleaning
Once my file is loaded, I check if there are duplicate values in any of the column that could affect the analysis. I am also looking for inconsistencies in values, data type or missing values that may affect the investigation.


```py
# PassengerId and Name must be unique. I check if there are any duplicate values in each col.
# There are no duplicates on the data.
print titanic_df.duplicated('PassengerId').sum()
print titanic_df.duplicated('Name').sum()
# I also check if the Ticket # is unique. Turns out that the Ticket isn't unique for each passenger.
# Seems odd and I will take note and come back to it if needed.
print titanic_df.duplicated('Ticket').sum()
```

{% highlight py %}

    0
    0
    210

{% endhighlight %}

```python
# I check the data type of each column for any inconsistencies. Seems odd to have Age as an float64.
# I investigate and print out a couple of rows with non-whole number age.
titanic_df.dtypes

    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object

# Saw 7 entries that are less than 1. Looking at their names, I see prefix "Master",
# which is what is given to children. In these case, these were babies below the age of 1.
non_whole = titanic_df['Age'] < 1
titanic_df[non_whole]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>79</td>
      <td>1</td>
      <td>2</td>
      <td>Caldwell, Master. Alden Gates</td>
      <td>male</td>
      <td>0.83</td>
      <td>0</td>
      <td>2</td>
      <td>248738</td>
      <td>29.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>305</th>
      <td>306</td>
      <td>1</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.92</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
    </tr>
    <tr>
      <th>469</th>
      <td>470</td>
      <td>1</td>
      <td>3</td>
      <td>Baclini, Miss. Helene Barbara</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>2666</td>
      <td>19.2583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>644</th>
      <td>645</td>
      <td>1</td>
      <td>3</td>
      <td>Baclini, Miss. Eugenie</td>
      <td>female</td>
      <td>0.75</td>
      <td>2</td>
      <td>1</td>
      <td>2666</td>
      <td>19.2583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>755</th>
      <td>756</td>
      <td>1</td>
      <td>2</td>
      <td>Hamalainen, Master. Viljo</td>
      <td>male</td>
      <td>0.67</td>
      <td>1</td>
      <td>1</td>
      <td>250649</td>
      <td>14.5000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>803</th>
      <td>804</td>
      <td>1</td>
      <td>3</td>
      <td>Thomas, Master. Assad Alexander</td>
      <td>male</td>
      <td>0.42</td>
      <td>0</td>
      <td>1</td>
      <td>2625</td>
      <td>8.5167</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>831</th>
      <td>832</td>
      <td>1</td>
      <td>2</td>
      <td>Richards, Master. George Sibley</td>
      <td>male</td>
      <td>0.83</td>
      <td>1</td>
      <td>1</td>
      <td>29106</td>
      <td>18.7500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I check the head of the data set.
titanic_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# and the tail.
titanic_df.tail()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Looking at Age col, I see that there are some empty fields. I check how many there are.
missing_age = titanic_df['Age'].isnull()
print 'There are {} logs with their Age not specified.'.format(missing_age.sum())
```

    There are 177 logs with their Age not specified.


## Exploratory
### Women and Children First
The 177 logs that have missing age will affect the analysis. I attempt to limit the descrepency by rephrasing my question to distinguish the survivability between women and children vs Male adult passengers.


```python
titanic_df['womChil'] = 0 # Created a column that groups women and children.
women = titanic_df['Sex'] == 'female' # Criteria - all female passengers.
child = titanic_df['Age'] < 19 # Criteria - all children under the age of 18 years.
# For passengers with missing age, I identify the children from the group of male passengers by looking
# for the title 'Master' in their Names, which are titles given to minors on board without their
# parents.
masters = titanic_df['Name'].str.contains('Master') # Criteria - all male children.
titanic_df.loc[(women | child | masters),'womChil'] = 1 # Add 1 (yes) that fits the criterias
women_children = titanic_df['womChil'] == 1 # Criteria - all Women and Children.
survived = titanic_df['Survived'] == 1 # Criteria - all passengers that survived.

# Extracted all women and children that sruvived.
women_children_survived = titanic_df[women_children & survived]
women_children_survived['PassengerId'].count() # Counted the women and children survivors.
print "Of the {} passengers that survived from the sample of 891 on the data provided, {} are \
women and children.".format(titanic_df[survived]['PassengerId'].count(), \
                            women_children_survived['PassengerId'].count())
```

    Of the 342 passengers that survived from the sample of 891 on the data provided, 259 are women and children.



```python
# Knowing that 259 of the passengers fit the criteria of women and children, I double check on
# the remaining survivors that did not fit the criteria and whose age were missing from the records
# and try find any other clue that might distinguish the the male passengers as children.
non_womChil = titanic_df['womChil'] == 0
print titanic_df[non_womChil & survived & missing_age]['Age'].isnull().sum()
titanic_df[non_womChil & survived & missing_age].sort_values('Pclass')
```

    14





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>womChil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>Woolner, Mr. Hugh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>19947</td>
      <td>35.5000</td>
      <td>C52</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>299</td>
      <td>1</td>
      <td>1</td>
      <td>Saalfeld, Mr. Adolphe</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>19988</td>
      <td>30.5000</td>
      <td>C106</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>507</th>
      <td>508</td>
      <td>1</td>
      <td>1</td>
      <td>Bradley, Mr. George ("George Arthur Brayton")</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>111427</td>
      <td>26.5500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>740</th>
      <td>741</td>
      <td>1</td>
      <td>1</td>
      <td>Hawksford, Mr. Walter James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>16988</td>
      <td>30.0000</td>
      <td>D45</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>839</th>
      <td>840</td>
      <td>1</td>
      <td>1</td>
      <td>Marechal, Mr. Pierre</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>11774</td>
      <td>29.7000</td>
      <td>C47</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>547</th>
      <td>548</td>
      <td>1</td>
      <td>2</td>
      <td>Padro y Manent, Mr. Julian</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>SC/PARIS 2146</td>
      <td>13.8625</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>1</td>
      <td>3</td>
      <td>Mamee, Mr. Hanna</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2677</td>
      <td>7.2292</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>108</td>
      <td>1</td>
      <td>3</td>
      <td>Moss, Mr. Albert Johan</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>312991</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>302</td>
      <td>1</td>
      <td>3</td>
      <td>McCoy, Mr. Bernard</td>
      <td>male</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>367226</td>
      <td>23.2500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>0</td>
    </tr>
    <tr>
      <th>444</th>
      <td>445</td>
      <td>1</td>
      <td>3</td>
      <td>Johannesen-Bratthammer, Mr. Bernt</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>65306</td>
      <td>8.1125</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>643</th>
      <td>644</td>
      <td>1</td>
      <td>3</td>
      <td>Foo, Mr. Choong</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1601</td>
      <td>56.4958</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>692</th>
      <td>693</td>
      <td>1</td>
      <td>3</td>
      <td>Lam, Mr. Ali</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>1601</td>
      <td>56.4958</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>828</th>
      <td>829</td>
      <td>1</td>
      <td>3</td>
      <td>McCormack, Mr. Thomas Joseph</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>367228</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def survivability(data, criteria):
    yes = data[criteria & survived]['PassengerId'].size
    whole = data[criteria]['PassengerId'].size
    chances = (float(yes) / float(whole))* 100
    return chances
```


```python
survive_womenAndChild = survivability(titanic_df, women_children)
survive_men = survivability(titanic_df, non_womChil)
```


```python
## There is no more criteria we can use to identify the remaining surviving male passengers with
## no specified age as children. We then move on and draw our conclusions.
ax = sns.countplot(x='Survived', hue='womChil', data=titanic_df)
sns.plt.show()

print "Women and Children had a {0:.2f}% chance of surviving the tragedy".format(survive_womenAndChild)
print "Men only had a {0:.2f}% chances of surviving the tragedy".format(survive_men)

```


![png](https://raw.githubusercontent.com/cmd-master/dand-titanic-survivors/master/img/output_16_0.png)


    Women and Children had a 66.58% chance of surviving the tragedy
    Men only had a 16.53% chances of surviving the tragedy


### Preferential Treatment
 We move on to the next question, would rich people have a higher chances of surviving the tragedy than the rest of the people on board Titanic?


```python
# To answer the question, we group the passengers according to Pclass and see the describe info.
# There were 216 Passengers in Upper Class. Their Fare price has a maximum value of 512.33.
# There are 184 Passengers in the Middle Class. Max Fare is 73.50
# There are 491 Passengers in the Lowest Class. Max Fare is 69.50
passengers_by_class = titanic_df.groupby(['Pclass'])
passengers_by_class.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>womChil</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">1</th>
      <th>count</th>
      <td>186.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.233441</td>
      <td>84.154687</td>
      <td>0.356481</td>
      <td>461.597222</td>
      <td>0.416667</td>
      <td>0.629630</td>
      <td>0.458333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.802856</td>
      <td>78.380373</td>
      <td>0.693997</td>
      <td>246.737616</td>
      <td>0.611898</td>
      <td>0.484026</td>
      <td>0.499418</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.920000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>30.923950</td>
      <td>0.000000</td>
      <td>270.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>60.287500</td>
      <td>0.000000</td>
      <td>472.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>93.500000</td>
      <td>0.000000</td>
      <td>670.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>4.000000</td>
      <td>890.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">2</th>
      <th>count</th>
      <td>173.000000</td>
      <td>184.000000</td>
      <td>184.000000</td>
      <td>184.000000</td>
      <td>184.000000</td>
      <td>184.000000</td>
      <td>184.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.877630</td>
      <td>20.662183</td>
      <td>0.380435</td>
      <td>445.956522</td>
      <td>0.402174</td>
      <td>0.472826</td>
      <td>0.494565</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.001077</td>
      <td>13.417399</td>
      <td>0.690963</td>
      <td>250.852161</td>
      <td>0.601633</td>
      <td>0.500623</td>
      <td>0.501335</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.670000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>23.000000</td>
      <td>13.000000</td>
      <td>0.000000</td>
      <td>234.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.000000</td>
      <td>14.250000</td>
      <td>0.000000</td>
      <td>435.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>36.000000</td>
      <td>26.000000</td>
      <td>1.000000</td>
      <td>668.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>70.000000</td>
      <td>73.500000</td>
      <td>3.000000</td>
      <td>887.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">3</th>
      <th>count</th>
      <td>355.000000</td>
      <td>491.000000</td>
      <td>491.000000</td>
      <td>491.000000</td>
      <td>491.000000</td>
      <td>491.000000</td>
      <td>491.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.140620</td>
      <td>13.675550</td>
      <td>0.393075</td>
      <td>439.154786</td>
      <td>0.615071</td>
      <td>0.242363</td>
      <td>0.405295</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.495398</td>
      <td>11.778142</td>
      <td>0.888861</td>
      <td>264.441453</td>
      <td>1.374883</td>
      <td>0.428949</td>
      <td>0.491450</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.000000</td>
      <td>7.750000</td>
      <td>0.000000</td>
      <td>200.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.000000</td>
      <td>8.050000</td>
      <td>0.000000</td>
      <td>432.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>32.000000</td>
      <td>15.500000</td>
      <td>0.000000</td>
      <td>666.500000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>74.000000</td>
      <td>69.550000</td>
      <td>6.000000</td>
      <td>891.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Seems odd that on the 3 classes, the minimum Fare is 0, which is free.
# I check the Fare priced at 0.
free = titanic_df['Fare'] == 0
freeloaders = titanic_df[free]
freeloaders # Looks like some of these are crew of the titanic but I can't be sure.
# Might be too complicated to include the Fare in my analysis. I ignore it and move on.
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>womChil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0</td>
      <td>B94</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0</td>
      <td>A36</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0</td>
      <td>B102</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I create criteria for each of the class and see how many survived for each.
first = titanic_df['Pclass'] == 1
second = titanic_df['Pclass'] == 2
third = titanic_df['Pclass'] == 3
non_survived = titanic_df['Survived'] == 0
```


```python
survive_first = survivability(titanic_df, first)
survive_second = survivability(titanic_df, second)
survive_third = survivability(titanic_df, third)
print "{0:.2f}% of the First Class Passengers survived.".format(survive_first)
print "{0:.2f}% of the Second Class Passengers survived.".format(survive_second)
print "{0:.2f}% of the Third Class Passengers survived.".format(survive_third)
```

    62.96% of the First Class Passengers survived.
    47.28% of the Second Class Passengers survived.
    24.24% of the Third Class Passengers survived.



```python
ax = sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
ax.set_xticklabels(['First', 'Second', 'Third'])

sns.plt.show()
print "You have a {0:.2f}% chance of surviving the tragedy if you were a First Class Passenger," \
.format(survive_first)
print "than if you were in 3rd Class, who only had a {0:.2f}% chance of surviving." \
.format(survive_third)
```


![png](https://github.com/cmd-master/dand-titanic-survivors/blob/master/img/output_22_0.png)


    You have a 62.96% chance of surviving the tragedy if you were a First Class Passenger,
    than if you were in 3rd Class, who only had a 24.24% chance of surviving.



```python
g = sns.barplot(x="womChil", y="Survived", hue="Pclass", data=titanic_df)
sns.plt.show()
print "If we look at the average count of women and children survivors for each class, it seems that, \
in passenger class, women and children had a higher chances of survival if you were in the upper class \
than if you were in the lower class."
```


![png](https://raw.githubusercontent.com/cmd-master/dand-titanic-survivors/master/img/output_23_0.png)


    If we look at the average count of women and children survivors for each class, it seems that, in passenger class, women and children had a higher chances of survival if you were in the upper class than if you were in the lower class.



```python
# I took a look on the combination of Age grouped according to their Economic Status
# into account and see the probability of Survivability.
fig = sns.boxplot(x='Pclass', y='Age', data=titanic_df, hue='Survived')
plt.show()
print "The Box Plot shows that people on First Class are generally much older than the lower class. \
Though it goes to show that much younger passengers survived the tragedy, the variability amongst \
the classes are greated the higher the class. On average the Younger Survivors belonged on the lower \
class and more older passengers belonged on the upper class."
```


![png](https://raw.githubusercontent.com/cmd-master/dand-titanic-survivors/master/img/output_24_0.png)


    The Box Plot shows that people on First Class are generally much older than the lower class. Though it goes to show that much younger passengers survived the tragedy, the variability amongst the classes are greated the higher the class. On average the Younger Survivors belonged on the lower class and more older passengers belonged on the upper class.


## Conclusions
Looking back on the questions that were asked at the beginning, we can observe certain
relations towards survivability of passengers with their Demographics and Socia Economic.


### "Women and Children First". Does your age or gender influence your chances of survivability?

With the data provided, it seems that women and children have a higher chances of suriving the tragedy than that of adult males. 67% of the women and children survived the tragedy while only 17% of the adult males survived. Please take note that analysis is limited to the fact that some of the passenger's age were missing in the dataset and that it may skew results.


### If you are rich, would you most likely be prioritized?

The answer to this question is most probably. Looking at the data grouped according to the Passenger Class, it would seem that the higher you are in the passenger class, the higher your chances of surviving the tragedy. 63% of the  First Class Passengers survived the tragedy. While only 48% of the Second Passengers survived and only 24% of the Third Class Passengers survived. Please take note that this part of the analysis is limited to the fact that there is no direct relationship between socio-economic status and their class. In other words, we cannot be sure that First Class passengers are rich and the rest of the passengers are poor. It would be great to have more details about each passenger such as their occupation or income for example. I looked at the Ticket Fare, but it seems that some passengers had 0 Fare, which could mean many things hence could not be a great basis in this part of the analysis.
