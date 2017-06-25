
# coding: utf-8

# In[ ]:

# Data Source: https://www.kaggle.com/worldbank/world-development-indicators
# Folder: 'world-development-indicators'


# <br><p style="font-family: Arial; font-size:3.75em;color:purple; font-style:bold">
# Matplotlib: Exploring <br> <br> <br>Data Visualization</p><br><br>

# <br><br><center><h1 style="font-size:2em;color:#2467C0">World Development Indicators</h1></center>
# <br>
# <table>
# <col width="550">
# <col width="450">
# <tr>
# <td><img src="https://upload.wikimedia.org/wikipedia/commons/4/46/North_South_divide.svg" align="middle" style="width:550px;height:360px;"/></td>
# <td>
# This week, we will be using an open dataset from <a href="https://www.kaggle.com">Kaggle</a>. It is  <a href="https://www.kaggle.com/worldbank/world-development-indicators">The World Development Indicators</a> dataset obtained from the World Bank containing over a thousand annual indicators of economic development from hundreds of countries around the world.
# <br>
# <br>
# This is a slightly modified version of the original dataset from <a href="http://data.worldbank.org/data-catalog/world-development-indicators">The World Bank</a>
# <br>
# <br>
# List of the <a href="https://www.kaggle.com/benhamner/d/worldbank/world-development-indicators/indicators-in-data">available indicators</a> and a <a href="https://www.kaggle.com/benhamner/d/worldbank/world-development-indicators/countries-in-the-wdi-data">list of the available countries</a>.
# </td>
# </tr>
# </table>

# # Step 1: Initial exploration of the Dataset

# In[1]:

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


# In[3]:

data = pd.read_csv('./world-development-indicators/Indicators.csv')
data.shape


# This is a really large dataset, at least in terms of the number of rows.  But with 6 columns, what does this hold?

# In[4]:

data.head(10)


# Looks like it has different indicators for different countries with the year and value of the indicator. 

# ### How many UNIQUE country names are there ?

# In[7]:

countries = data['CountryName'].unique()
len(countries)


# ### Are there same number of country codes ?

# In[8]:

# How many unique country codes are there ? (should be the same #)
countryCodes = data['CountryCode'].unique().tolist()
len(countryCodes)


# ### Are there many indicators or few ?

# In[9]:

# How many unique indicators are there ? (should be the same #)
indicators = data['IndicatorName'].unique().tolist()
len(indicators)


# ### How many years of data do we have ?

# In[10]:

# How many years of data do we have ?
years = data['Year'].unique().tolist()
len(years)


# ### What's the range of years?

# In[11]:

print(min(years)," to ",max(years))


# <p style="font-family: Arial; font-size:2.5em;color:blue; font-style:bold">
# Matplotlib: Basic Plotting, Part 1</p><br>

# ### Lets pick a country and an indicator to explore: CO2 Emissions per capita and the USA

# In[46]:

# select CO2 emissions for India
hist_indicator = 'CO2 emissions \(metric'
hist_country = 'IND'

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['CountryCode'].str.contains(hist_country)

# stage is just those indicators matching the USA for country code and CO2 emissions over time.
stage = data[mask1 & mask2]

# select CO2 emissions for the United States
hist_indicator = 'CO2 emissions \(metric'
hist_country = 'USA'

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['CountryCode'].str.contains(hist_country)

# stageU is just those indicators matching the USA for country code and CO2 emissions over time.
stageU = data[mask1 & mask2]


# In[47]:

stage.head()


# In[48]:

stageU.head()


# ### Let's see how emissions have changed over time using MatplotLib

# In[49]:

# get the years
years = stage['Year'].values
# get the values 
co2 = stage['Value'].values

# create
plt.bar(years,co2)
plt.show()


# In[50]:

# switch to a line plot
plt.plot(stage['Year'].values, stage['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])

#label the figure
plt.title('CO2 Emissions in India')

# to make more honest, start they y axis at 0
plt.axis([1959, 2011,0,2])

plt.show()

# switch to a line plot
plt.plot(stage['Year'].values, stage['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])

#label the figure
plt.title('CO2 Emissions in India')

# to make more honest, start they y axis at 0
plt.axis([1959, 2011,0,25])

plt.show()

# switch to a line plot
plt.plot(stageU['Year'].values, stageU['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(stageU['IndicatorName'].iloc[0])

#label the figure
plt.title('CO2 Emissions in USA')

# to make more honest, start they y axis at 0
plt.axis([1959, 2011,0,25])

plt.show()


# ### Using Histograms to explore the distribution of values
# We could also visualize this data as a histogram to better explore the ranges of values in CO2 production per year. 

# In[55]:

# If you want to just include those within one standard deviation fo the mean, you could do the following
# lower = stage['Value'].mean() - stage['Value'].std()
# upper = stage['Value'].mean() + stage['Value'].std()
# hist_data = [x for x in stage[:10000]['Value'] if x>lower and x<upper ]

# Otherwise, let's look at all the data
hist_data = stage['Value'].values


# In[56]:

print(len(hist_data))


# In[57]:

# the histogram of the data
plt.hist(hist_data, 10, normed=False, facecolor='green')

plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Years')
plt.title('Histogram Example')

plt.grid(True)

plt.show()


# So India has many years where it produced between 0.3-0.4 metric tons per capita with outliers on either side.

# ### But how do the USA's numbers relate to those of other countries?

# In[58]:

# select CO2 emissions for all countries in 2011
hist_indicator = 'CO2 emissions \(metric'
hist_year = 2011

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['Year'].isin([hist_year])

# apply our mask
co2_2011 = data[mask1 & mask2]
co2_2011.head()


# For how many countries do we have CO2 per capita emissions data in 2011

# In[59]:

print(len(co2_2011))


# In[28]:

# let's plot a histogram of the emmissions per capita by country

# subplots returns a touple with the figure, axis attributes.
fig, ax = plt.subplots()

ax.annotate("USA",
            xy=(18, 5), xycoords='data',
            xytext=(18, 30), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

plt.hist(co2_2011['Value'], 10, normed=False, facecolor='green')

plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Countries')
plt.title('Histogram of CO2 Emissions Per Capita')

#plt.axis([10, 22, 0, 14])
plt.grid(True)

plt.show()


# So the USA, at ~18 CO2 emissions (metric tons per capital) is quite high among all countries.
# 
# An interesting next step, which we'll save for you, would be to explore how this relates to other industrialized nations and to look at the outliers with those values in the 40s!

# <p style="font-family: Arial; font-size:2.0em;color:blue; font-style:bold">
# Matplotlib: Basic Plotting, Part 2</p>

# ### Relationship between GPD and CO2 Emissions in USA

# In[74]:

# select GDP Per capita emissions for the United States
hist_indicator = 'GDP per capita \(constant 2005'
hist_country = 'IND'

mask1 = data['IndicatorName'].str.contains(hist_indicator) 
mask2 = data['CountryCode'].str.contains(hist_country)

# stage is just those indicators matching the USA for country code and CO2 emissions over time.
gdp_stage = data[mask1 & mask2]

#plot gdp_stage vs stage


# In[75]:

gdp_stage.head(2)


# In[63]:

stage.head(2)


# In[77]:

# switch to a line plot
plt.plot(gdp_stage['Year'].values, gdp_stage['Value'].values)

# Label the axes
plt.xlabel('Year')
plt.ylabel(gdp_stage['IndicatorName'].iloc[0])

#label the figure
plt.title('GDP Per Capita India')

# to make more honest, start they y axis at 0
#plt.axis([1959, 2011,0,47000])

plt.show()


# So although we've seen a decline in the CO2 emissions per capita, it does not seem to translate to a decline in GDP per capita

# ### ScatterPlot for comparing GDP against CO2 emissions (per capita)
# 
# First, we'll need to make sure we're looking at the same time frames

# In[78]:

print("GDP Min Year = ", gdp_stage['Year'].min(), "max: ", gdp_stage['Year'].max())
print("CO2 Min Year = ", stage['Year'].min(), "max: ", stage['Year'].max())


# We have 3 extra years of GDP data, so let's trim those off so the scatterplot has equal length arrays to compare (this is actually required by scatterplot)

# In[79]:

gdp_stage_trunc = gdp_stage[gdp_stage['Year'] < 2012]
print(len(gdp_stage_trunc))
print(len(stage))


# In[80]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.set_title('CO2 Emissions vs. GDP \(per capita\)',fontsize=10)
axis.set_xlabel(gdp_stage_trunc['IndicatorName'].iloc[0],fontsize=10)
axis.set_ylabel(stage['IndicatorName'].iloc[0],fontsize=10)

X = gdp_stage_trunc['Value']
Y = stage['Value']

axis.scatter(X, Y)
plt.show()


# In[81]:

np.corrcoef(gdp_stage_trunc['Value'],stage['Value'])


# ## Want more ? 
# 
# ### Matplotlib Examples Library

# http://matplotlib.org/examples/index.html

# In[ ]:

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

