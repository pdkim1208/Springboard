
# coding: utf-8

# In[106]:


import pandas as pd
import json
from pandas.io.json import json_normalize

data = 'world_bank_projects.json'
json_df = pd.read_json(data)

most_common_country = json_df['countryname'].value_counts()

print(most_common_country.head(10))



# In[91]:


df2 = json_df['mjtheme_namecode']

values = {}
for i in df2:
    for j in i:
        if j['name'] in values and j['name'] != '':
            values[j['name']] += 1
        elif j['name'] == '':
            next
        else:
            values[j['name']] = 1
            
values_emptystring = {}
for i in df2:
    for j in i:
        if j['name'] in values_emptystring:
            values_emptystring[j['name']] += 1
        else:
            values_emptystring[j['name']] = 1

#Printing values without empty string
print(sorted(values, key=values.get, reverse=True)[:10])

#Printing values with empty string
print(sorted(values_emptystring, key=values_emptystring.get, reverse=True)[:10])


# In[ ]:


proper_df = json_df.copy()

# Empty dictionary to hold key-vals, 'code':'name'
dict_values = {}
for i in df2:
    for j in i:
        if j['name'] == '':
            next  
        elif j['code'] in dict_values:
            next
        else:
            dict_values[j['code']] = j['name']
            
for i in proper_df['mjtheme_namecode']:
    for j in i:
        if j['name'] == '':
            j['name'] = dict_values[j['code']]
        else:
            next

print(proper_df['mjtheme_namecode'])

