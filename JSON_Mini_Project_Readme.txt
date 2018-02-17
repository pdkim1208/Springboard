The purpose of this exercise is to demonstrate the ability to work with JSON files, in particular, to extract data from specific columns and to demonstrate the ability to find the most common values in a certain column.

We begin by reading the JSON file, and I found the “countryname” column by using the using the .columns method (not included in the code). A simple value_counts() method call reveals the answer to the first question.


In the second block of code, I used a double for loop to extract each dictionary individually, as each row consists of a list of one or more dictionaries. In doing so, we can use the .groupby() function, then use the value_counts() method to answer exercise 2.

In the third block of code, I started out by creating a lookup table. I did so by returning all the rows with non-blank values for ‘name’, and then dropping the duplicates. Then I set the index as the column ‘code’ and converted it into an integer, as it was previously an object. We can verify the accuracy of the lookup table by calling lookup_table[‘name’].nuniques(), which is 11, and matches new_list3[‘name’].nuniques() from exercise 2. The for loop that I wrote in the third block goes into each dictionary in a copied dataframe (json_df4) and assigns the value from the lookup table only if the value associated with ‘name’ is blank. We can verify that this has worked because we know that the second dictionary in the first row of the dataset was an empty string (‘’), and now it is filled in with the correct value.

