# Mall_Customer-Segmentation

## The Challenges:
*The dataset is a customer database of a mall. It contains 200 observations with basic information such as age, gender, annual income, and spending score. The purpose of this analysis is to uncover underlying patterns in the customer base, and to groups of customers accordingly, often known as market segmentation. In doing so, the marketing team can have a more targeted approach to reach consumers, and the mall can make more informed strategic decisions to increase profits.*

## Data:
This project is a part of the Mall Customer Segmentation Data competition held on Kaggle.
The dataset can be downloaded from the kaggle website which can be found https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python.

## Apporach:
Understanding the data.

Data analysis and EDA

Visualization

Building model and Training

Perform elbow method to find optimal no. of clustering

Plotting the cluster 

Plotting the 3d model .

Conclusions


## Understanding the data:
 #### **Basic understanding**:
 The most common ways in which businesses segment their customer base are:

   ###### ***Demographic information*** : 
  such as gender, age, familial and marital status, income, education, and occupation.
   
   ##### ***Geographical information***:
   which differs depending on the scope of the company. For localized businesses, this info might pertain to specific towns or counties. For larger                       companies, it might mean a customer’s city, state, or even country of residence.
  
  ##### ***Psychographics***: 
   such as social class, lifestyle, and personality traits.
   
   #### ***Behavioral data***: 
   such as spending and consumption habits, product/service usage, and desired benefits
   
  #### **Advantages of Customer Segmentation**:
  > Price Optimization
   
  > Enhances competitivness
   
  > Brand Awarness
 
  > Acquistion and Retention
 
  > Increase Revenue and ROI
  
 ## **Main Understanding**:
 ***The main components of this project are exploratory data analysis and model building***.
 ### libraries required :
 
 ##### Pandas
 
 ##### Numpy
 
 ##### Matplotlib
 
 ##### Seaborn
 
 ##### Scikit-learn
 
## Data Cleaning and Exploratory Data Analysis:
#### In this section we are doing a little bit of data exploration, checking for null values, object data types and other things we might consider in order to keep our data clean and well structured 


![1](https://user-images.githubusercontent.com/114009434/195463002-8527013f-397e-4664-aeeb-468e1ccd1c19.jpg)
##### Here we have the following features :

######  1. CustomerID: ***It is the unique ID given to a customer***
###### 2. Gender: ***Gender of the customer***
###### 3. Age: ***The age of the customer***
###### 4. Annual Income(k$): ***It is the annual income of the customer***
###### 5. Spending Score: ***It is the score(out of 100) given to a customer by the mall authorities, based on the money spent and the behavior of the customer.***

#### ***We have zero null values in any column***

### Statistics Summary of Dataset:
![0](https://user-images.githubusercontent.com/114009434/195467808-67aa6074-1317-4701-8f00-7aef445c39a4.png)
###### Age: ***min age: 18 , max age: 70***
###### Aunnal Income : ***min : 15K$ and max : 137K$***
###### Spending Score : ***min : 01 and max : 99***


## VISUALIZATION

### Gender Basis:
![3](https://user-images.githubusercontent.com/114009434/195567049-1ce5f94e-e993-46f8-8428-8a5d3b3e6aa1.jpg)

***from data we can say that ratio of male is less than female.***

***male= 44%***
***female = 56%***
![image](https://user-images.githubusercontent.com/114009434/195606453-8b521a58-21d4-4102-b29d-fb53db0508d9.png)


###### **Now it's the moment to visualize our data and plot important information so we can see the different values our data has and its behaviour. To do so, I am going to consider the following features: Annual_income, Spending_score and Age. Gender will only be used to make data sepparation so I can differentiate values for men and women.To begin with, I am plotting the histograms for each of the three features we said we would look into:**
![005](https://user-images.githubusercontent.com/114009434/195571686-d7a3351d-2bce-476c-b611-153bd29be1f5.png) ![006](https://user-images.githubusercontent.com/114009434/195571773-6cc99f12-9122-4a4b-985c-e2dfcf9a8b7e.jpg) ![007](https://user-images.githubusercontent.com/114009434/195571835-1a8c395f-dee6-4f3b-8660-bbde485d58b5.jpg)
##### **In these histograms I observed that the distribution of these values resembles a Gaussian distribution, where the vast majority of the values lay in the middle with some exceptions in the extremes.**

###  Visualization distribution of number of customers in each age group: 
##### To find out the range of age which have highest number of customer . therefore from statistics summary of dataset i got a reading where minimun age is 18 and maximum age is 70. so i divided the age into 5 group that is age from 18-25 , 26-35,36-45,45-55 and 56+.
![1](https://user-images.githubusercontent.com/114009434/195574985-053fdecc-abc6-443d-82db-e5f5dacdd718.png)
##### **Clearly the 26–35 age group outweighs every other age group**.

###  visualization distribution of the customers score according to their spending scores.
##### To find out majority of the customer having highest score ,i consiodered the  statistics summary of dataset i got a reading where minimun spending score is 01 and maximum spending score is 99. so i divided the spending score  into 5 group that is age from 1-20 , 21-40,41-60,61-80 and 81-100.
![2](https://user-images.githubusercontent.com/114009434/195576116-9980b03c-12cd-4a80-a6c0-b203ae7a7094.png)
##### **The majority of the customers have spending score in the range 41–60.**

###  visualization distribution of highest income customers according to their annual income.
##### To find out customer income from their annual income, i consider the  statistics summary of dataset where i got a reading where minimun annual salary is 15k$ and maximum annual salary  is 137k$. so i divided the annual income  into 5 group that is age from 1-30k$ , 31-60k$,61-90k$,91-120k$ and 121-150k$.
![3](https://user-images.githubusercontent.com/114009434/195580121-102c3679-43e6-4afc-8563-b3315d1a0f36.png)
##### **The majority of the customers have annual income in the range 61k$ and 90k$.**

### Building the Model:
##### **Standardizing data is a good practice when clustering, as the range of values within each feature will influence how the cluster is formed, which is not usually desirable. Kmeans clustering uses Euclidean distance to measure the similarity between objects, so if a feature has a range much larger than another feature, it will dominate the other features in the clustering process.The model was initiated on the standardized data to cluster based on age, income, and spending score.**

### Tuning the Model :
When using Kmeans, the number of clusters (k), is a value to be set by the user. There are a few methods to determine the appropriate number of clusters, as shown below
### Elbow Method:

##### Calculate the Within Cluster Sum of Squared Errors (WSS) for different values of k, and choose the k for which WSS first starts to diminish. In the plot of   WSS-versus k, this is visible as an elbow.The steps can be summarized in the below steps:
1)***Compute K-Means clustering for different values of K by varying K from 1 to 10 clusters.***

2)***For each K, calculate the total within-cluster sum of square (WCSS).***

3)***Plot the curve of WCSS vs the number of clusters K.***

4)***The location of a bend (knee) in the plot is generally considered as an indicator of the appropriate number of clusters.***

***Within Cluster Sum Of Squares (WCSS) against the the number of clusters (K Value) to figure out the optimal number of clusters value. WCSS measures sum of distances of observations from their cluster centroids which is given by the below formula.***
      ![image](https://user-images.githubusercontent.com/114009434/195611347-cf63b257-787f-4d33-84fe-2039c32e5b27.png)
 
 ***where Yi is centroid for observation Xi. The main goal is to maximize number of clusters and in limiting case each data point becomes its own cluster centroid.***
 
 ## clustering relationship between age and spending score:
 ![image](https://user-images.githubusercontent.com/114009434/195612135-cf19f0a2-31f9-420e-8c07-16040e513048.png)

***It is clear from the figure that we should take the number of clusters equal to 4.***

**After getting value of k . now plotting the cluster of the customer on the basis age vs spending score.**
![image](https://user-images.githubusercontent.com/114009434/195618832-44388864-91f1-4123-b799-bc345b9538df.png)

**Analyzing the Results:**

***1)We can see that the mall customers can be broadly grouped into 4 groups based on their purchases made in the mall***
        
        **cluster 1: Blue**
        **cluster 2: voilet**
        **cluster 3: Red**
        **cluster 4: grey**

***2)Each cluster having centriod which is in brown colour***

***3)cluster 1 having highest spending score.***

***4)cluster 2 having moderate spending score but cluster 3 customer age belong to younger than cluster 4.***

***5)cluster 4 having least spending score.***

## Clustering relationship between Annual income and Spending score:
![image](https://user-images.githubusercontent.com/114009434/195627995-26de25f5-60b2-4a8b-88ca-c7a381c9a120.png)

***It is clear from the figure that we should take the number of clusters equal to 5.***

**After getting value of k . now plotting the cluster of the customer on the income vs spending score.**
![image](https://user-images.githubusercontent.com/114009434/195628775-dc655fe1-9349-44f8-b472-9225be37267a.png)

**Analyzing the Results:**

***1)We can see that the mall customers can be broadly grouped into 5 groups based on their purchases made in the mall***
        
        **cluster 1: Green**
        **cluster 2: Orange**
        **cluster 3: Red**
        **cluster 4: Violet**
        **cluster 5: Blue**


***2)In cluster 1(green-colored) we see that people have high income and high spending scores, this is the ideal case for the mall or shops as these people are the prime sources of profit. These people might be the regular customers of the mall and are convinced by the mall’s facilities.***

***3)In cluster 2(orange colored) we can see that people have low income but higher spending scores, these are those people who for some reason love to buy products more often even though they have a low income. Maybe it’s because these people are more than satisfied with the mall services. The shops/malls might not target these people that effectively but still will not lose them.***

***4)In cluster 3(red colored) we see that people have high income but low spending scores, this is interesting. Maybe these are the people who are unsatisfied or unhappy by the mall’s services. These can be the prime targets of the mall, as they have the potential to spend money. So, the mall authorities will try to add new
facilities so that they can attract these people and can meet their needs.***

***5)In cluster 4(violet colored) we can see people have low annual income and low spending scores, this is quite reasonable as people having low salaries prefer to buy less, in fact, these are the wise people who know how to spend and save money. The shops/mall will be least interested in people belonging to this cluster.***

***6)In cluster 5(pink colored) we see that people have average income and an average spending score, these people again will not be the prime targets of the shops or mall, but again they will be considered and other data analysis techniques may be used to increase their spending score.***


## Clustering relationship between Age ,Annual income and Spending score:
![image](https://user-images.githubusercontent.com/114009434/195635710-bd5f33bc-093c-481f-987a-92f0ea83fbc1.png)

***It is clear from the figure that we should take the number of clusters equal to 5.***

**After getting value of k . now plotting the cluster of the customer on the age vs annual income vs spending score.**
![image](https://user-images.githubusercontent.com/114009434/195636397-0001c79e-ad9f-46b7-9fde-51767d037219.png)

## Conclusions:

***1)KMeans Clustering is a powerful technique in order to achieve a decent customer segmentation.***

***2)Customer segmentation is a good way to understand the behaviour of different customers and plan a good marketing strategy accordingly.***

***3)There isn't much difference between the spending score of women and men, which leads us to think that our behaviour when it comes to shopping is pretty similar.***

***4)Observing the clustering graphic, it can be clearly observed that the ones who spend more money in malls are young people. That is to say they are the main target when it comes to marketing, so doing deeper studies about what they are interested in may lead to higher profits.***

***5)Althought younglings seem to be the ones spending the most, we can't forget there are more people we have to consider, like people who belong to the pink cluster, they are what we would commonly name after "middle class" and it seems to be the biggest cluster.***

***6)Promoting discounts on some shops can be something of interest to those who don't actually spend a lot and they may end up spending more***

 
 
 
