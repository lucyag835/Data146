{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "editorial-guide",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>country</th>\n",
       "      <th>continent</th>\n",
       "      <th>year</th>\n",
       "      <th>lifeExp</th>\n",
       "      <th>pop</th>\n",
       "      <th>gdpPercap</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1952</td>\n",
       "      <td>28.801</td>\n",
       "      <td>8425333</td>\n",
       "      <td>779.445314</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1957</td>\n",
       "      <td>30.332</td>\n",
       "      <td>9240934</td>\n",
       "      <td>820.853030</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1962</td>\n",
       "      <td>31.997</td>\n",
       "      <td>10267083</td>\n",
       "      <td>853.100710</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1967</td>\n",
       "      <td>34.020</td>\n",
       "      <td>11537966</td>\n",
       "      <td>836.197138</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Afghanistan</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1972</td>\n",
       "      <td>36.088</td>\n",
       "      <td>13079460</td>\n",
       "      <td>739.981106</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1675</th>\n",
       "      <td>Yemen, Rep.</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1987</td>\n",
       "      <td>52.922</td>\n",
       "      <td>11219340</td>\n",
       "      <td>1971.741538</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1676</th>\n",
       "      <td>Yemen, Rep.</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1992</td>\n",
       "      <td>55.599</td>\n",
       "      <td>13367997</td>\n",
       "      <td>1879.496673</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1677</th>\n",
       "      <td>Yemen, Rep.</td>\n",
       "      <td>Asia</td>\n",
       "      <td>1997</td>\n",
       "      <td>58.020</td>\n",
       "      <td>15826497</td>\n",
       "      <td>2117.484526</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1678</th>\n",
       "      <td>Yemen, Rep.</td>\n",
       "      <td>Asia</td>\n",
       "      <td>2002</td>\n",
       "      <td>60.308</td>\n",
       "      <td>18701257</td>\n",
       "      <td>2234.820827</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1679</th>\n",
       "      <td>Yemen, Rep.</td>\n",
       "      <td>Asia</td>\n",
       "      <td>2007</td>\n",
       "      <td>62.698</td>\n",
       "      <td>22211743</td>\n",
       "      <td>2280.769906</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>396 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          country continent  year  lifeExp       pop    gdpPercap\n",
       "0     Afghanistan      Asia  1952   28.801   8425333   779.445314\n",
       "1     Afghanistan      Asia  1957   30.332   9240934   820.853030\n",
       "2     Afghanistan      Asia  1962   31.997  10267083   853.100710\n",
       "3     Afghanistan      Asia  1967   34.020  11537966   836.197138\n",
       "4     Afghanistan      Asia  1972   36.088  13079460   739.981106\n",
       "...           ...       ...   ...      ...       ...          ...\n",
       "1675  Yemen, Rep.      Asia  1987   52.922  11219340  1971.741538\n",
       "1676  Yemen, Rep.      Asia  1992   55.599  13367997  1879.496673\n",
       "1677  Yemen, Rep.      Asia  1997   58.020  15826497  2117.484526\n",
       "1678  Yemen, Rep.      Asia  2002   60.308  18701257  2234.820827\n",
       "1679  Yemen, Rep.      Asia  2007   62.698  22211743  2280.769906\n",
       "\n",
       "[396 rows x 6 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Read file\n",
    "import pandas as pd\n",
    "path_to_data = 'gapminder.tsv'\n",
    "data = pd.read_csv(path_to_data,sep='\\t')\n",
    "\n",
    "#Largest GDP per capita\n",
    "max_gdp = data['gdpPercap'].max()\n",
    "max_gdp\n",
    "max_gdp_idx = data['gdpPercap'].idxmax()\n",
    "max_gdp_idx\n",
    "data.loc[max_gdp_idx]\n",
    "\n",
    "idx_asia = data['continent']=='Asia'\n",
    "data_asia = data[idx_asia]\n",
    "data_asia.loc[11]\n",
    "data_asia.iloc[0]\n",
    "data_asia"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "dangerous-adelaide",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1952, 1987, 1957, 1992, 1962, 1997, 1967, 2002, 1972, 2007, 1977, 1982]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Exercises\n",
    "#All the years in the data, no duplicates\n",
    "years = list(set(list(data['year'])))\n",
    "years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "architectural-continent",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#How many unique values are there?\n",
    "len(years)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "gentle-michigan",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[1952, 1987, 1957, 1992, 1962, 1997, 1967, 2002, 1972, 2007, 1977, 1982]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#What are they?\n",
    "years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "polar-bachelor",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1318683096"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#What is the largest pop value?\n",
    "pop_max = data['pop'].max()\n",
    "pop_max"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "continuous-occasion",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2007"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#When and where did this occur\n",
    "idx_pop_max = data['pop'].idxmax()\n",
    "data.loc[idx_pop_max]['year']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "answering-toner",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'China'"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.loc[idx_pop_max]['country']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "perfect-toolbox",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Extract records for Europe\n",
    "idx_europe = data['continent'] == 'Europe'\n",
    "data_europe = data[idx_europe]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "mechanical-lesbian",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Iceland'"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#In 1952, which country had the smallest population?\n",
    "idx_europe_1952 = (data['continent']=='Europe') & (data['year']==1952)\n",
    "data_europe_1952 = data[idx_europe_1952]\n",
    "pop_min = data_europe_1952['pop'].min()\n",
    "pop_min\n",
    "idx_pop_min = data_europe['pop'].idxmin()\n",
    "data.loc[idx_pop_min]['country']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "detailed-charm",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "695    301931\n",
       "Name: pop, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#What was the population in 2007?\n",
    "idx_iceland_2007 = (data['year']==2007) & (data['country']=='Iceland')\n",
    "data_iceland_2007 = data[idx_iceland_2007]\n",
    "data_iceland_2007['pop']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "confused-trailer",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "695    301931\n",
       "Name: pop, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#In one step:\n",
    "iceland_pop_2007 = data[(data['year']==2007)&(data['country']=='Iceland')]['pop']\n",
    "iceland_pop_2007"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  },
  "toc-autonumbering": true
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
