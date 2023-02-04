# OTTO – Multi-Objective Recommender System
This is the my code for the Kaggle OTTO - Multi-Objective Recommender System challenge in 2022/2023. I finished the competition in the top 2% of participants, coming 52nd out of 2615 teams. The dataset is quite large, especially during later stages of the pipeline so a high ram (200gb+) google cloud virtual machine was used for the later scripts. 

## Brief competition overview: 
The goal of the competition is to predict e-commerce clicks, cart additions, and orders from real data provided by Germany's largest online store OTTO. We are given the transaction history of a customer truncated at some point in time, and need to create a model that predicts what items they will click on, add to their cart, and ultimately order. These models enhance retailers' ability to predict which products each customer actually wants to see, add to their cart, and order at any given moment of their visit leading to a more personalised online shopping experience. 

## My solution:
A common approach to this type of problem is that of the two stage recommender, namely:
- An initial model generates a large list of candidates for each user, based on their previously seen shopping behaviour. 
- A second model then refines this large list of candidates into a much smaller subset. This usually involves calculating a number of features for each candidate item (which would be too computationally expensive to do for all items, hence step 1!) and then using these features to rank the items by the probability that they are what the customer wants. 

A summary of the various technical methods used to complete both step 1 and 2 above is shown below: 
![plot](/docs/pipeline_outline.PNG)

## Optimisations 
- As mentioned above the code was ran on a high memory virtual machine. 
- The feature generation script is written in polars rather than pandas since polars has a much smaller memory footprint, and lead to significant speed ups both in calculations but especialy when joining large dataframes. 
- Most of the functions for loading the datasets allow for subsets to be sampled, this was important for pipeline development. 

## More information
A lot more technical details such as how accuracy improved over time, which features worked best etc. is avaliable in my full [Kaggle write up](https://pages.github.com/).
