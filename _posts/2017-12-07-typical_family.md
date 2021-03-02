---
layout: post
date: 2017-12-07 5:47
title:  "Typycal Filipino Family"
mood: happy
category:
- udacity
---

# Typical Filipino

Filipinos are harding working people and yet they tend to be careless with
their money. They would work for long hours, sometimes far from home, and
end up spending all their money on stuff they don't need. It is common to hear
stories of Filipinos spending a lifetime working overseas and coming back
poorer than they were before the left.
<!--more-->

## Objectives
In this article, I would like to know how Filipinos spend their money in
relation to their income.

## Dataset Source
This dataset is from The Philippine Statistics Authority who conducts the
Family Income and Expediture Survey (FEIS) every 3 years nationwide. This is
from the 2015 most recent survey. The raw data was cleaned by Francis Paul
Flores from his [kaggle.com/grosvenpaul](https://www.kaggle.com/grosvenpaul/family-income-and-expenditure/downloads/Family%20Income%20and%20Expenditure.csv).

# Analysis Section
## Univiriate Plots Section
The Dataset contains 41,544 observations of Filipino Households from every
Region of the country. It is comprised of 60 variables describing each family
on their income, family description and expenditure.


```
##  [1] "Total Household Income"                       
##  [2] "Region"                                       
##  [3] "Total Food Expenditure"                       
##  [4] "Main Source of Income"                        
##  [5] "Agricultural Household indicator"             
##  [6] "Bread and Cereals Expenditure"                
##  [7] "Total Rice Expenditure"                       
##  [8] "Meat Expenditure"                             
##  [9] "Total Fish and  marine products Expenditure"  
## [10] "Fruit Expenditure"                            
## [11] "Vegetables Expenditure"                       
## [12] "Restaurant and hotels Expenditure"            
## [13] "Alcoholic Beverages Expenditure"              
## [14] "Tobacco Expenditure"                          
## [15] "Clothing, Footwear and Other Wear Expenditure"
## [16] "Housing and water Expenditure"                
## [17] "Imputed House Rental Value"                   
## [18] "Medical Care Expenditure"                     
## [19] "Transportation Expenditure"                   
## [20] "Communication Expenditure"                    
## [21] "Education Expenditure"                        
## [22] "Miscellaneous Goods and Services Expenditure"
## [23] "Special Occasions Expenditure"                
## [24] "Crop Farming and Gardening expenses"          
## [25] "Total Income from Entrepreneurial Acitivites"
## [26] "Household Head Sex"                           
## [27] "Household Head Age"                           
## [28] "Household Head Marital Status"                
## [29] "Household Head Highest Grade Completed"       
## [30] "Household Head Job or Business Indicator"     
## [31] "Household Head Occupation"                    
## [32] "Household Head Class of Worker"               
## [33] "Type of Household"                            
## [34] "Total Number of Family members"               
## [35] "Members with age less than 5 year old"        
## [36] "Members with age 5 - 17 years old"            
## [37] "Total number of family members employed"      
## [38] "Type of Building/House"                       
## [39] "Type of Roof"                                 
## [40] "Type of Walls"                                
## [41] "House Floor Area"                             
## [42] "House Age"                                    
## [43] "Number of bedrooms"                           
## [44] "Tenure Status"                                
## [45] "Toilet Facilities"                            
## [46] "Electricity"                                  
## [47] "Main Source of Water Supply"                  
## [48] "Number of Television"                         
## [49] "Number of CD/VCD/DVD"                         
## [50] "Number of Component/Stereo set"               
## [51] "Number of Refrigerator/Freezer"               
## [52] "Number of Washing Machine"                    
## [53] "Number of Airconditioner"                     
## [54] "Number of Car, Jeep, Van"                     
## [55] "Number of Landline/wireless telephones"       
## [56] "Number of Cellular phone"                     
## [57] "Number of Personal Computer"                  
## [58] "Number of Stove with Oven/Gas Range"          
## [59] "Number of Motorized Banca"                    
## [60] "Number of Motorcycle/Tricycle"
```

Each variable is described accordingly in their labels.


```
##  [1] "income"          "region"          "expense"        
##  [4] "source"          "agri_house"      "exp_bread"      
##  [7] "exp_rice"        "exp_meat"        "exp_seafood"    
## [10] "exp_fruit"       "exp_veg"         "exp_resto_hotel"
## [13] "exp_alcoh"       "exp_taba"        "exp_clothe"     
## [16] "exp_house_water" "exp_rent"        "exp_med"        
## [19] "exp_trans"       "exp_comms"       "exp_edu"        
## [22] "exp_misc"        "exp_spec"        "exp_farm"       
## [25] "inc_entrep"      "head_gender"     "head_age"       
## [28] "head_stat"       "head_educ"       "head_job_bus"   
## [31] "head_occup"      "head_workclass"  "family_t"       
## [34] "family_n"        "baby_n"          "kid_n"          
## [37] "employed_n"      "house_t"         "roof_t"         
## [40] "wall_t"          "house_area"      "house_age"      
## [43] "bed_n"           "house_tenure"    "toilet"         
## [46] "electric"        "water_t"         "tv_n"           
## [49] "DVD_n"           "sterio_n"        "ref_n"          
## [52] "wash_n"          "aircon_n"        "car_n"          
## [55] "tel_n"           "cell_n"          "pc_n"           
## [58] "stove_n"         "mboat_n"         "mbike_n"
```

I have modified each one to make it easier for coding.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

There are a lot of low-income and few high-income households that would make
the graph skew to the left and stretch out to the right.


```
##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
##    11285   104895   164080   247556   291138 11815988
```

Average income per household would be at 247,556 Php annually.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

I wanted to see where this average is in our distribution. Since the
distribution is skewed by the large disparity of income, I transform the income
into log base 10. Think of it like I am bringing the incomes bins closer
together to get a better view on the distribution.

Marked in red, the average income is slightly on the right from the center.
This tells us that the average does not best describe the whole distribution.
In other words, 247k Php annual income is really high for most Filipinos.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

I took a closer look at the head_workclass or the working class for each job
occupation and see their counts.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

I also check the ages of the heads fo the family.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

In addition, I checked the number of family members in each households.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

I divide the distribution according to the working class as defined by the
survey. I do this so that I can see the relationship of the population average
income with the different groups of working classes.

This looks better but the categories are a bit vague and are not familiar of
how I see our working class in society.


```r
library(stringr)
library(dplyr)

class.expert <- paste(c("managers", "medical", "dentist", "lawyer",
                     "architect", "supervisor", "trade", "education",
                     "chemist", "professionals", "scientist",
                     "justice", "Mechanical engineers", "director",
                     "accountant", "Chemical engineers",
                     "Electrical Engineers", "Industrial engineers",
                     "designer", "singers", "television", "dancer",
                     "fashion", "marketing", "Civil engineers", "air",
                     "Computer programmers", "nurse", "Librarians",
                     "science", "therapist", "opticians", "Veter",
                     "pilots", "estate agents", "plant operators",
                     "legislative"),
                   collapse = "|")

class.farmer <- paste(c("farm", "fish", "growers", "gather", "duck",
                      "cultivat", "plant", "hunter", "village",
                      "practitioners", "healers", "diary", "animal"),
                    collapse = "|")

class.worker <- paste(c("workers", "carpenters", "welder", "helper", "clean",
                      "launder", "labor", "driver", "freight", "mechanics",
                      "mason", "porter", "waiter", "cook", "foremen",
                      "caretaker", "lineman", "varnish", "conductor",
                      "salesperson", "bookkeeper", "inspector", "undertaker",
                      "loggers", "wood", "roofers", "cutter", "electricians",
                      "assembler", "builder", "metal",  "tanners", "garbage",
                      "repair", "prepare", "rigger", "vendors", "valuers",
                      "setters", "guides", "tasters", "potters", "preservers",
                      "textile", "fitters", "valets", "blasters",
                      "humanitarian", "staff officers"),
                    collapse = "|")

class.office <- paste(c("pawn", "buy", "baker", "maker", "tailor", "assistant",
                     "clerk", "engineering technicians", "operators",
                     "artists", "managers/managing proprietors", "insurance",
                     "educ", "principals", "secretaries", "general",
                     "pharmacists", "commercial", "communications engineers",
                     "draftsmen", "instructors", "travel consultants",
                     "enlisted personnel"),
                    collapse = "|")

fies$head_class <- with(fies,
                  ifelse(str_detect(head_workclass, "government"),
                         "gov",
                  ifelse(grepl(class.worker, head_occup, ignore.case = T),
                         "work",
                  ifelse(grepl(class.office, head_occup, ignore.case = T),
                         "off",
                  ifelse(grepl(class.farmer, head_occup, ignore.case = T),
                         "farm",
                  ifelse(grepl(class.expert, head_occup, ignore.case = T),
                         "exp",
                         NA))))))
```



Therefore, I create our own classification of the working class based each on
the occupation of the head of the family.


```
## # A tibble: 1 x 1
##       mean
##      <dbl>
## 1 239113.9
```

Since I am ignoring households that did not specify the occupation of their
head, I get the average income of households that did and store it in inc.ave.


```
## $exp
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   25133  321260  530172  668838  843932 4208400
##
## $farm
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   12911   83998  119494  160549  178824 4810822
##
## $gov
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   24886  174310  323632  426544  558545 3805717
##
## $off
##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
##    21136   134373   216450   317292   369152 11815988
##
## $work
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   11285  104172  151013  193944  232346 7082152
```

I take a quick summary of the different classes and see the different means of
each of the working class.


```
## # A tibble: 14,459 x 3
##                                                                     head_occup
##                                                                         <fctr>
##  1                                                     Street ambulant vendors
##  2                                           Market and sidewalk stall vendors
##  3                                          Waiters, waitresses and bartenders
##  4 General managers/managing proprietors in personal care, cleaning and relati
##  5                                         Shop salespersons and demonstrators
##  6                                           Market and sidewalk stall vendors
##  7                                           Market and sidewalk stall vendors
##  8                                                         Building caretakers
##  9                                  Production supervisors and general foremen
## 10                                         Shop salespersons and demonstrators
## # ... with 14,449 more rows, and 2 more variables: income <int>,
## #   inc_entrep <int>
```

This is where I group each working class according the occupation of the head
of the family starting with the worker class.


```
## # A tibble: 1 x 3
##                           head_occup  income inc_entrep
##                               <fctr>   <int>      <int>
## 1 Waiters, waitresses and bartenders 5652261    5107451
```

I noticed one outlier under the worker class that was interesting. One Waiter
was earning Php5,652,261 annually. This does not seem right to me so I took a
deeper look and found out that *income* is actually the sum of salary income
and income from their private business.

The Waiter was just earning Php544,810 from his job and the rest was from his
business, which was Php5,107,451. I reviewed my groupings again and made sure
that I was grouping them according to their income salary from their occupation
and not from their overall income.


```
##
##   0 137 150 540 550 667
##  22   1   1   1   1   1
```

I created a new variable that only included their salary income. If income is
the overall income, then deducting income from their business (inc_entrep)
should not give us a negative value.


```
## $exp
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##    4339  295431  490000  602606  761157 3310612
##
## $farm
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##    1212   26991   54926   89381  103539 3034500
##
## $gov
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   18370  147436  294358  387665  518604 3210933
##
## $off
##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
##        0    52901   120242   194908   246241 11639365
##
## $work
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##       0   85926  130510  166069  203123 1796730
```


```
## $exp
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   25133  321260  530172  668838  843932 4208400
##
## $farm
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   12911   83998  119494  160549  178824 4810822
##
## $gov
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   24886  174310  323632  426544  558545 3805717
##
## $off
##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
##    21136   134373   216450   317292   369152 11815988
##
## $work
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
##   11285  104172  151013  193944  232346 7082152
```

I will still use the mean of their overall income for my basis of the analysis.
However, I will be using their salary income in grouping them according to
their respective working classes.


```
## # A tibble: 6 x 4
##                                         head_occup income_mean
##                                             <fctr>       <dbl>
## 1               Agronomists and related scientists     2155298
## 2 Aircraft pilots, navigators and flight engineers     2089931
## 3                                          Lawyers     1295705
## 4   Directors and chief executives of corporations     1271361
## 5                               Chemical engineers     1207281
## 6                                         Justices     1201001
## # ... with 2 more variables: income_median <dbl>, n <int>
```


```
## # A tibble: 6 x 4
##                                                 head_occup income_mean
##                                                     <fctr>       <dbl>
## 1                         Power production plant operators   1013486.5
## 2 Incinerator, water treatment and related plant operators    942367.5
## 3    Glass and ceramics kiln and related machine operators    780429.0
## 4                                        School principals    733687.4
## 5           Technical and commercial sales representatives    606841.5
## 6                                Insurance representatives    580991.7
## # ... with 2 more variables: income_median <dbl>, n <int>
```


```
## # A tibble: 6 x 4
##                                                                    head_occup
##                                                                        <fctr>
## 1                                                        Other animal raisers
## 2 Production and operations managers in agriculture, hunting, forestry and fi
## 3                                                            Tree nut farmers
## 4                                                        Hunters and trappers
## 5                                                       Other poultry farmers
## 6                                    Fish-farm cultivators (excluding prawns)
## # ... with 3 more variables: income_mean <dbl>, income_median <dbl>,
## #   n <int>
```

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-22-1.png)<!-- -->

In our dataset, I generally have more workers than any of the other classes.
Experts are the minority of the group.

## Univariate Analysis

### What is the structure of your dataset?
There are 41,544 observations in the data set with 60 variables. Description of
each variables are described on the column names from the raw data. The
first set of variables decribes the income and expenses of each household. The
income consists of the salary of the head of the family and income from their
private businesses. The expenses are broken down into more details such as food
expenditure, housing, luxury items, etc. The second set of variables describes
the family and head of the family, whose income provides the greater share of
the income. It discusses the age, gender, working class and occupation of the
head of the family. As well as, the some characteristics of the family itself
like number of family members, number of children and number of employed in the
family. The third set of variables are the properties that the family own. This
describes the type of house they have, the number of cars, phones, tv, etc.

### What is/are the main feature(s) of interest in your dataset?
I am particularly interested in the income, occupation of the head of the
family and their expenses. I want to see how spending habits change as income
increases and what each family prioritize in spending.

### Did you create any new variables from existing variables in the dataset?
As I analyze Income in the next section, I found out that Income includes
income from salaries and their private businesses. Once I discovered this, I
subtracted business income from overall income to get the salary income. I also
derived the savings, total expenses, savings to income ratio as I go along the
analysis.

### Did you perform Data Wrangling?
The Working Class as defined by the survey is vague and unfamiliar to me. As a
result, I evaluated each occupation of the head of the family and categorized
each household under a new working class variable called *head_class*.

The working classes that I have defined are as follows:

- *Expert.* Jobs that usually requires higher education.
- *Office* Usually found in shops and offices.
- *Government.* Jobs in government.
- *Workers.* Jobs that usually does not require a degree.
- *Farmers.* Jobs that pertain to rural work. Agriculture, Fishing, etc.

These categories are based on my own classifications of how I see the working
class and their overall mean income. I am ignoring the households that did not
specify the occupation of the head of the family.


## Bivariate Plots Section
### Income Source

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-23-1.png)<!-- -->

I go back to the overall income distribution but this time using our custom
work class. I add the average income Php239k and I see that some groups
are below the average line and other groups are above the average line.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-24-1.png)<!-- -->

If people relied from their salaries, the picture would generally look the same
except for the office and workers, where salaries are more spread out to the
left.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-25-1.png)<!-- -->

If people relied solely on income from their business, all of them are below
the Average line of the overall income population.

These grouped histograms shows that Filipinos who works in offices and workers
would commonly rely mostly from their salaries but would still have income from
private business that would support their overall income.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-26-1.png)<!-- -->

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-27-1.png)<!-- -->

Both graphs shows that experts are heavily dependent on salaries. While most
groups would treat their business as a secondary source of income, office
employees and farmers would seem to have more profitable business than the rest
of the groups.


```
## # A tibble: 68,016 x 4
##       id head_class   inc_type    inc
##    <int>      <ord>      <chr>  <int>
##  1     1        gov   inc_work 435962
##  2     1        gov inc_entrep  44370
##  3     2       work   inc_work 198235
##  4     2       work inc_entrep      0
##  5     3       work   inc_work  82785
##  6     3       work inc_entrep      0
##  7     4       farm   inc_work  92009
##  8     4       farm inc_entrep  15580
##  9     5        off   inc_work 113635
## 10     5        off inc_entrep  75687
## # ... with 68,006 more rows
```

I would like to compare work income from business income to find out the
relationship. I transform the data from wide format to long format so that
I can graph the data in a frequency polygon.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-29-1.png)<!-- -->

I see that farmers have both their salaries from business income at almost
similar distribution with their salary income. Office Employees have income
from Salaries higher than their business income. While workers would have a
distribution that have a higher income from Salaries and income from business
more spread out. Government employees and Professionals does not have a clear
relationship between income from salaries business.

But this is not clear because I am mixing households with 2 sources of income
(salary and Business) with the ones with only 1 source of income.


```
## # A tibble: 6 x 4
##      id head_class inc_type       prop
##   <int>      <ord>    <chr>      <dbl>
## 1     1        gov   salary 0.90762639
## 2     1        gov business 0.09237361
## 3     4       farm   salary 0.85518966
## 4     4       farm business 0.14481034
## 5     5        off   salary 0.60022079
## 6     5        off business 0.39977921
```

```
## [1] 23897
```

Out of the total 41,544 households, 23,897 have 2 sources of income. I extract
these observations and mold the data to enable us to see where these income are
coming from according to our work class.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-32-1.png)<!-- -->

Households who have 2 sources income would rely more on their salary than they
would their business. This is more so for Experts, Government and Workers where
they would get 75% of their income from their salary. While Office and Farmers
would have their Business income producing at almost the same level from their
Salary income.

## Bivariate Analysis
### How did the feature(s) of interest vary with other features in the dataset?
I was particularly interested on how income from salary related to to their
income from their businesses. Some households would be industrious enough to
have work for a company and at the same time have a side business. I wanted to
know how much of their income was from their salary and how much was from their
business. I found out that although many households have a business on the
side, they will still always be dependent on salaries.

### What was the strongest relationship you found?
The strongest relationship is between income and expense. It is an obvious fact
that as income increase so does their expenses. If that is the case, I wanted
to know what do Filipino households spend their income on. In other words, what
are their financial priorities and if this changes as income increase?

## Multivariate Analysis

### Spending Habits




![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-34-1.png)<!-- -->

I want to relate expense with their income and if I graph the relationship,
I see that as income increase, so does their expenses, which is obvious.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-35-1.png)<!-- -->

The same graph but this time with color grouped according to our custom
I see which work class have higher incomes. But this is not really clear
because of overplotting.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-36-1.png)<!-- -->

This graph shows as a more linear relationship between expense and income but
the working class is still unclear.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-37-1.png)<!-- -->

Splitting the graphs in a facet, it seems that all classes are resembling the
same pattern where the more income they have the higher they would spend. This
does not tell us anything new. And I get to thinking, what are they actually
spending on?


```
## # A tibble: 272,064 x 4
##       id head_class  cashflow        cash
##    <int>      <ord>    <fctr>       <dbl>
##  1     1        gov      Food 0.398710508
##  2     1        gov    Luxury 0.042562427
##  3     1        gov      Vice 0.000000000
##  4     1        gov     House 0.269154817
##  5     1        gov   Medical 0.009937078
##  6     1        gov   Farming 0.055678679
##  7     1        gov Education 0.104056179
##  8     1        gov    Living 0.119900313
##  9     2       work      Food 0.322843155
## 10     2       work    Luxury 0.056804834
## # ... with 272,054 more rows
```

I just realized that the *expense* is the total expense of food and not the
total expense overall. The above graphs in this section are wrong. I correct
this by adding all expenses under *expense_total*.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-39-1.png)<!-- -->

This boxplots shows the proportion of each group expenses. It enables us to see
what each goup are spending on and are prioritizing. I have grouped the
expenses according to the following categories:

- *Food:* Bread, Cereals, Rice, Meat, Fish, Fruit, Vegetables Expenses.
- *Luxury:* Restaurant, Hotel, Clothing and Special Occasions Expenses.
- *Vices:* Alcoholic and Tabacco Expenses.
- *House:* Housing, Water, Rent Expenses.
- *Living:* Transportation, Communication, Misc and Services Expenses.
- *Medical:* Medical Expenses.
- *Education:* Education Expenses.
- *Farming:* Farming and Gardening Expenses.



I add the expenses overall and add it to our main table *fies* as *expense*.


```
## # A tibble: 272,064 x 5
##       id head_class income  cashflow   cash
##    <int>      <ord>  <int>     <chr>  <int>
##  1     1        gov 480332      Food 138707
##  2     1        gov 480332    Luxury  14807
##  3     1        gov 480332      Vice      0
##  4     1        gov 480332     House  93636
##  5     1        gov 480332   Medical   3457
##  6     1        gov 480332   Farming  19370
##  7     1        gov 480332 Education  36200
##  8     1        gov 480332    Living  41712
##  9     2       work 198235      Food  68712
## 10     2       work 198235    Luxury  12090
## # ... with 272,054 more rows
```

I transform the data into another long format but this time expenses are split
according to their different types. Income will be used as reference bu when I
do transform the data, it replicates income 8 times. It does this because there
are 8 types of expenses. I tried chaning the income as a factor but it made my
graph into a qualitative graph that made my x-axis lables to detailed. This
just makes the points darker but the plot is still more or less the same.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-42-1.png)<!-- -->

From the graph, it would seem Filipinos would tend to prioritize food above
else regardless of working class. With the exception of the farmers, who would
have farming as their major expense.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-43-1.png)<!-- -->

I look at the inverse of the graph and see the data from the perspective of the
type of expenses. It is still a little bit chaotic but there are certain
aspects I can see. For instance, housing, living and luxury expenses are
linearly correlated. As income increases, these expenses increase as well. Food
is interesting because it also increases but only to a point. I gues no matter
how rich you are, your taste in food does not change.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-44-1.png)<!-- -->

Splitting the data in a grid, I see that housing, living and luxury have a
clear linear relationship with income. It seems that these expenses increases
as income increases ragardless of the working class. Food expense increases
only to a point. It is sad to see that education seems to be as varied as their
expenses for vices.

### Savings

I take a look at the savings of each Filipino households by taking in a new
variable *savings*, which is income minus expense.


```r
fies <- transform(fies, savings = income - expense)
fies <- transform(fies, sav_inc_ratio = savings/income)
```

Once I have the Savings of each household, I divide savings over income to get
the ratio. I was surprised to see many households in debt.


```
##
## deficit surplus
##     313   41231
```

I counted how many were in debt and those who had a surplus and found that it
was not that many. Only 313 out ouf 41,544 households were in debt and who were
living beyond their means. It is only 0.75 % of the population.


```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
## -8.5995  0.4396  0.5624  0.5510  0.6824  0.9843
```

With the summary of savings to income ration, I see that most Filipinos would
save half of their income.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-48-1.png)<!-- -->

It is good to know that most Filipinos would save their money rather than
spending it.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-49-1.png)<!-- -->

I facet the same graph according to the working class and see a similar trend
when graphing income.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-50-1.png)<!-- -->

Savings and income in a scatterpolt allows me to see a linear trend for all
working classes.

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-51-1.png)<!-- -->

It seems that savings proportion is consistent in that as income increases, so
does their proportion of savings.

## Mutlivariate Analysis
### Were there features that strengthened each other?
It is interesting to discover that the ratio between income from business and
from salaries varies in the working class. Experts, Government and Workers
would be heavily dependent on their Salaries while Office employees and Farmers
have their sources of income split, with salaries as their main source of
income.

### Were there any surprising interactions between features?
As I plot the relationship between income and their expenses, it is surprising
to see that the relationship between food expense and income gradually weakens
as income increases. And it is overtaken by housing expenses. I guess their
food preference do not change as their income increases but their ambitions of
having a bigger house continually changes. This is something that requires
further analysis.

## Final Plots

### Side Business

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-52-1.png)<!-- -->

Filipino households would tend to have 2 sources of income, either from their
salaries or from their side businesses. Experts, workers and government would
rely mostly on their salaries. While 55% of the income of office employees and
farmers would be from their salaries, 45% would be from their side business.
This is interesting because it shows that office workers and farmers are more
motivated to have a second source of income than the rest of the working class.
Or maybe they have more extra time in their hands  to have a business on the
side to earn more income.

### Financial Priorities

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-53-1.png)<!-- -->

Filipinos would generally prioritize Food, Housing, Living and Luxry items
over everything else. Lower income households would have Food (Green) as their
top priority but, as income grows, it levels off and housing expense slowly
takes over as their most expensive priority. I guess their food preferences and
cost does not change as income increases.

### Filipino Savings

![](https://raw.githubusercontent.com/angelocandari/eda-filipino-family/master/eda-filipino-family_files/figure-html/unnamed-chunk-54-1.png)<!-- -->

Filipinos would tend to keep 56% or almost half of their income to savings. But
savings proportions varies according to different groups of working class.
Higher income jobs would generally save more proportion of their income than
lower income jobs.

## Reflections

I find it hard to go back and forth through the exploratory process. When I go
through the analysis, I find certain details that I have overlooked. At this
stage, I have to decide if I should go back to the beginning and change that
detail or should I change it at the point where I found the problem. Take for
example, I thought I was using the total expense in my analysis. It turns out
that I was using the total Food expense. I realized that I was using the wrong
expense after several graphs. I had to choices: to go back and change recompute
the total expense or change it at the stage of my analysis and move on. I ended
up doing the latter.

I guess the hardest part of doing exploratory data analysis (EDA) is
encountering doubt along the process. At some point, I start to question my own
way of thinking and I was afraid that I made too many assumptions, which makes
the whole process questionable. On the other hand, I had to be decisive and
trust my gut feeling so that I can move on. I think the best way to get through
it all is to have 3 principles in mind.

First, **to see things for what they are**. I try to see thinks as objectively
as I can. i try not to overthink a certain variable or what it means. Most of
all, I try to explain things as plainly as I can even to myself so that people,
including me, can understand the process.

Second, **I act on the things that I can control**. There are a ton of
variables that I can use and many variations on how to use them. Focusing on
the things that I can control can make things more easier.

Lastly, **to let go of things that I cannot control**. I cannot know or analyze
everything. I have to let of of things that I have no control over and trust
that everything will make sense in the end.

For future work and anyone who encounters this analysis, try not to rely on
averages and try to think plainly. The terms used in the survey, when I first
found it, was a bit vague and alien. For example, Working Class as defined by
the survey was too technical and did not really relate to a common sociatal
structure that we commonly see. It would be a lot useful if the data indicates
if the occupation is an overseas worker or is working within the country.
Because I know that a lot of Filipinos are working abroad. In addition, a
construction worker working in the Philippines has a much lower salary than the
a construction working working abroad.
