```
#Read file
import pandas as pd
path_to_data = 'gapminder.tsv'
(data = pd.read_csv(path_to_data,sep='\t'))

#Largest GDP per capita
max_gdp = data['gdpPercap'].max()
max_gdp
max_gdp_idx = data['gdpPercap'].idxmax()
max_gdp_idx
data.loc[max_gdp_idx]

idx_asia = data['continent']=='Asia'
data_asia = data[idx_asia]
data_asia.loc[11]
data_asia.iloc[0]

data_frame = data_frame.reset_index(drop=True)
data_asia
```
```
#Exercises
#All the years in the data, no duplicates
years = list(set(list(data['year'])))
```
```
#How many unique values are there?
len(years)
```
```
#What are they?
years
#What is the largest pop value?
pop_max = data['pop'].max()
pop_max
#When and where did this occur
idx_pop_max = data['pop'].idxmax()
data.loc[idx_pop_max]['year']
data.loc[idx_pop_max]['country']
#Extract records for Europe
idx_europe = data['continent'] == 'Europe'
data_europe = data[idx_europe]
#In 1952, which country had the smallest population?
idx_europe_1952 = (data['continent']=='Europe') & (data['year']==1952)
data_europe_1952 = data[idx_europe_1952]
pop_min = data_europe_1952['pop'].min()
pop_min
idx_pop_min = data_europe['pop'].idxmin()
data.loc[idx_pop_min]['country']
#What was the population in 2007?
idx_iceland_2007 = (data['year']==2007) & (data['country']=='Iceland')
data_iceland_2007 = data[idx_iceland_2007]
data_iceland_2007['pop']
#In one step:
iceland_pop_2007 = data[(data['year']==2007)&(data['country']=='Iceland')]['pop']
iceland_pop_2007
```
