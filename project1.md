# Project 1
# Lucy Greenman

## Part 1
### What is a package?
A package is a collection of modules, each of which can contain a set of related commands.

### What is a library?
A library is a collection of related functions that can be downloaded together for use in a work session.

### What two steps do you need to execute in order to install a package and then make that library of functions accessible?
First, you need to install the package: `pip install pandas`. Then, to make the library of functions accessible, you need to import the library: `import numpy as np`.

### Provide examples of how you would execute these two steps using packages from class.


```python
#Example 1
pip install numpy
import numpy as np
#Example 2
pip install pandas
import pandas as pd
```

### Be sure to include an alias and explain why it is a good idea to do so.
In the second example above, I typed not just "import pandas" but "import pandas as pd". This makes "pd" the alias for "pandas", a shorter name that I can use to call functions. Using an alias can save time and also reduce the likelihood of typos interfering with code running correctly.

## Part 2
### What is a data frame?
A data frame is a way of organizing data that consists of rows and columns. The first row is a header row containing column names, and the first column is an index by which the data frame is sorted.

### Identify a library of functions that is particularly useful for working with data frames.
A great one is pandas.
### In order to read a file in its remote location within the file system of your operating system, which command would you use?
pd.read_csv
### Provide an example of how to read a file and import it into your work session in order to create a new data frame.


```python
data = pd.read_csv('gapminder.tsv',sep='\t')
```

### Also, describe why specifying an argument within a read_() function can be significant.
The read_csv() function assumes comma-separated values, so if the values are separated otherwise (by tabs, in this case), then that must be specified.
### Does data that is saved as a file in a different type of format require a particular argument in order for a data frame to be successfully imported?
Yes, the argument sep must be defined if the values are separated by something other than commas.
### Also, provide an example that describes a data frame you created. How do you determine how many rows and columns are in a data frame? Is there an alternate terminology for describing rows and columns?
The number of rows and columns is given by the shape function. The size of the data, given by the size function, describes the number of rows multiplied by the number of columns. Additionally, a row can be called an observation and a column can be called a variable.


```python
data.shape
```




    (1704, 6)




```python
data.size
```




    10224



## Part 3
### Import the gapminder.tsv data set and create a new data frame.


```python
df = pd.read_csv('gapminder.tsv',sep='\t')
```

### Interrogate and describe the year variable within the data frame you created.


```python
df.year
```




    0       1952
    1       1957
    2       1962
    3       1967
    4       1972
            ... 
    1699    1987
    1700    1992
    1701    1997
    1702    2002
    1703    2007
    Name: year, Length: 1704, dtype: int64



### Does this variable exhibit regular intervals? If you were to add new outcomes to the raw data in order to update and make it more current, which years would you add to each subset of observations?
Yes, the variable occurs in intervals of 5. If I were adding years since 2007, I would add 2012 and 2017.

### Stretch goal: can you identify how many new outcomes in total you would be adding to your data frame?
For each year, data is added for every country. If I add two additional years, that will be an outcome for each year and for each country. Since there are 2 years and 142 countries, 284 new outcomes would be added in total.


```python
countries = len(df['country'].unique())
countries*2
```




    284



## Part 4
### Using the data frame you created by importing the gapminder.tsv data set, determine which country at what point in time had the lowest life expectancy.
Rwanda in 1992 had the lowest life expectancy.


```python
min_lifeExp = df['lifeExp'].min()
min_lifeExp
```




    23.599




```python
idx_min_lifeExp = (df['lifeExp'] == min_lifeExp)
temp = df[idx_min_lifeExp]
temp['country']
```




    1292    Rwanda
    Name: country, dtype: object




```python
temp['year']
```




    1292    1992
    Name: year, dtype: int64



### Conduct a cursory level investigation as to why this was the case and provide a brief explanation in support of your explanation.
In the 1990s, Rwanda was experiencing a genocide. Tens of thousands of Tutsis, an ethnic minority group, were murdered. This helps to explain the drastically low life expectancy of 23.599 years.

## Part 5
### Using the data frame you created by importing the gapminder.tsv data set, multiply the variable pop by the variable gdpPercap and assign the results to a newly created variable.


```python
gdp = df['pop']*df['gdpPercap']
df['gdp'] = gdp
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
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
      <th>gdp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
      <td>6.567086e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
      <td>30.332</td>
      <td>9240934</td>
      <td>820.853030</td>
      <td>7.585449e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
      <td>31.997</td>
      <td>10267083</td>
      <td>853.100710</td>
      <td>8.758856e+09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
      <td>34.020</td>
      <td>11537966</td>
      <td>836.197138</td>
      <td>9.648014e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
      <td>36.088</td>
      <td>13079460</td>
      <td>739.981106</td>
      <td>9.678553e+09</td>
    </tr>
  </tbody>
</table>
</div>



### Then subset and order from highest to lowest the results for Germany, France, Italy and Spain in 2007. Create a table that illustrates your results (you are welcome to either create a table in markdown or plot/save in PyCharm and upload the image).


```python
subset_country = df[df['country'].isin(['Germany','France','Italy','Spain'])]
subset_2007 = subset_country[subset_country['year']==2007]
subset_2007 = subset_2007.sort_values('gdp',ascending=False)
subset_2007
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
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
      <th>gdp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>575</th>
      <td>Germany</td>
      <td>Europe</td>
      <td>2007</td>
      <td>79.406</td>
      <td>82400996</td>
      <td>32170.37442</td>
      <td>2.650871e+12</td>
    </tr>
    <tr>
      <th>539</th>
      <td>France</td>
      <td>Europe</td>
      <td>2007</td>
      <td>80.657</td>
      <td>61083916</td>
      <td>30470.01670</td>
      <td>1.861228e+12</td>
    </tr>
    <tr>
      <th>779</th>
      <td>Italy</td>
      <td>Europe</td>
      <td>2007</td>
      <td>80.546</td>
      <td>58147733</td>
      <td>28569.71970</td>
      <td>1.661264e+12</td>
    </tr>
    <tr>
      <th>1427</th>
      <td>Spain</td>
      <td>Europe</td>
      <td>2007</td>
      <td>80.941</td>
      <td>40448191</td>
      <td>28821.06370</td>
      <td>1.165760e+12</td>
    </tr>
  </tbody>
</table>
</div>



Country | GDP
--- | ---
Germany | 2.65e+12
France | 1.86e+12
Italy | 1.66e+12
Spain | 1.17e+12

### Stretch goal: which of the four European countries exhibited the most significant increase in total gross domestic product during the previous 5-year period (to 2007)?
Germany's total gdp grew the most between 2002 and 2007.


```python
subset_country = df[df['country'].isin(['Germany','France','Italy','Spain'])]
subset_2002 = subset_country[subset_country['year']==2002]
subset_2002 = subset_2002.sort_values('gdp',ascending=False)
germanyDif = subset_2007.iloc[0]['gdp'].astype(int) - subset_2002.iloc[0]['gdp'].astype(int)
franceDif = subset_2007.iloc[1]['gdp'].astype(int) - subset_2002.iloc[1]['gdp'].astype(int)
italyDif = subset_2007.iloc[2]['gdp'].astype(int) - subset_2002.iloc[2]['gdp'].astype(int)
spainDif = subset_2007.iloc[3]['gdp'].astype(int) - subset_2002.iloc[3]['gdp'].astype(int)
max(germanyDif,franceDif,italyDif,spainDif)
germanyDif
```




    177402446824



## Part 6
### You have been introduced to four logical operators thus far: &, ==, | and ^. Describe each one including its purpose and function. Provide an example of how each might be used in the context of programming.
The & operator determines whether both conditions are true. For example, the expression ```(2>1) & (3>2)``` is true, because both sides of the comparison are true. == is used to evaluate whether two things are equal to each other. For example, ```2 == 2``` is true. | is the inclusive or, which evaluates to true if either condition is true or if both are true. For example, ```(2>1) | (2<1)``` would still evaluate to true even though the second condition is false. Yet ```(2>1) | (1<2)``` also evaluates to true, and both conditions are true. ^ is the exclusive or, which only evaluates to true if one but not both conditions are true. For example, ```(2>1) ^ (2<1)``` would still evaluate to true even though the second condition is false. But ```(2>1) ^ (1<2)``` also evaluates to false, since both conditions are true.


```python
(2>1) & (3>2)
```




    True




```python
2 == 2
```




    True




```python
(2>1) | (2<1)
```




    True




```python
(2>1) | (1<2)
```




    True




```python
(2>1) ^ (2<1)
```




    True




```python
(2>1) ^ (1<2)
```




    False



## Part 7
### Describe the difference between .loc and .iloc. Provide an example of how to extract a series of consecutive observations from a data frame.
.loc retrieves a value based on the label given by its index. .iloc retrieves a value based on its integer position in a series. Extracting a series of consecutive observations can be done with .iloc, by extracting the data from several integer positions in a row:


```python
df.iloc[0:5]
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
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
      <td>30.332</td>
      <td>9240934</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
      <td>31.997</td>
      <td>10267083</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
      <td>34.020</td>
      <td>11537966</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
      <td>36.088</td>
      <td>13079460</td>
      <td>739.981106</td>
    </tr>
  </tbody>
</table>
</div>



### Stretch goal: provide an example of how to extract all observations from a series of consecutive columns.
To extract observations from a series of consecutive columns, a subset of the data frame can be created using column names. iloc can also be used to subset consecutive columns by using a colon to indicate that all rows should be included and specifying which columns to subset in brackets.


```python
columns = ['country','continent','year']
df_subset = df[columns]
df_subset.head()
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
      <th>country</th>
      <th>continent</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:,[0,1,2]]
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
      <th>country</th>
      <th>continent</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1997</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2007</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 3 columns</p>
</div>



## Part 8
### Describe how an api works. Provide an example of how to construct a request to a remote server in order to pull data, write it to a local file and then import it to your current work session.
An API is an application programming interface, the part of a remote server that interacts with requests. It's basically the messenger between the client and the provider. To construct a request, you need to import the requests library and specify the url to be requested. Then a local file must be created for writing, such as in the format below. Finally, the file can be imported to the current work session as a pandas data frame.


```python
import requests
url = "https://api.covidtracking.com/v1/states/daily.csv"
r = requests.get(url)
file_name = 'thisismyfile' + '.csv'
with open(file_name, 'wb') as f:
    f.write(r.content)
    import pandas as pd
dfex = pd.read_csv(file_name)
dfex.head()
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
      <th>date</th>
      <th>state</th>
      <th>positive</th>
      <th>probableCases</th>
      <th>negative</th>
      <th>pending</th>
      <th>totalTestResultsSource</th>
      <th>totalTestResults</th>
      <th>hospitalizedCurrently</th>
      <th>hospitalizedCumulative</th>
      <th>...</th>
      <th>dataQualityGrade</th>
      <th>deathIncrease</th>
      <th>hospitalizedIncrease</th>
      <th>hash</th>
      <th>commercialScore</th>
      <th>negativeRegularScore</th>
      <th>negativeScore</th>
      <th>positiveScore</th>
      <th>score</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20210217</td>
      <td>AK</td>
      <td>54799.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>1611971.0</td>
      <td>32.0</td>
      <td>1238.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>1</td>
      <td>5</td>
      <td>c002b868e44913a74a70528d3fa75e7e3e5b4191</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20210217</td>
      <td>AL</td>
      <td>483167.0</td>
      <td>104382.0</td>
      <td>1859516.0</td>
      <td>NaN</td>
      <td>totalTestsPeopleViral</td>
      <td>2238301.0</td>
      <td>1030.0</td>
      <td>44541.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>89</td>
      <td>0</td>
      <td>3d4a8eef551e9cf3e5ab585388add5a45719e97b</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20210217</td>
      <td>AR</td>
      <td>314192.0</td>
      <td>65337.0</td>
      <td>2344576.0</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>2593431.0</td>
      <td>638.0</td>
      <td>14392.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>26</td>
      <td>0</td>
      <td>1de895ab8fa94a2d14163418fea02180332cea5f</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20210217</td>
      <td>AS</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2140.0</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>2140.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2857f0702fbfc1243e77e13a044b5830f3bea397</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20210217</td>
      <td>AZ</td>
      <td>801055.0</td>
      <td>53685.0</td>
      <td>2901236.0</td>
      <td>NaN</td>
      <td>totalTestsViral</td>
      <td>7281277.0</td>
      <td>1941.0</td>
      <td>55983.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>82</td>
      <td>118</td>
      <td>7f4d3ca6991968ec0fb4227de7d9ced82b0d1ccf</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>



## Part 9
### Describe the apply() function from the pandas library. What is its purpose?
The apply() function applies the same function to each row in a data frame. It can quickly and tidily enact the same action upon many rows of data.
### Using apply() to various class objects is an alternative (potentially preferable approach) to writing what other type of command? Why do you think apply() could be a preferred approach?
A loop can also be used to go through the data frame and apply a function to each row. apply() does this more neatly and efficiently.

## Part 10
### Also describe an alternative approach to filtering the number of columns in a data frame. Instead of using .iloc, what other approach might be used to select, filter and assign a subset number of variables to a new data frame?
Instead of using .iloc, a subset of a data frame can be created by using specific column names. The subset is selected by listing the column names, filtered by viewing the data frame of only those column names, and assigned to a new variable name (df_subset), as shown below.


```python
columns = ['country','continent','year']
df_subset = df[columns]
df_subset.head()
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
      <th>country</th>
      <th>continent</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
    </tr>
  </tbody>
</table>
</div>


