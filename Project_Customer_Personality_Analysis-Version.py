#!/usr/bin/env python
# coding: utf-8

# In[59]:


#Import Libraries
import pandas as pd
import numpy as np
import datetime

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Required models for modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import sys
np.random.seed(42)

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Loading the dataset
customerdata = pd.read_excel("/workspaces/Project-Customer-Personality-Analysis/marketing_campaign.xlsx")
print("Number of datapoints:", len(customerdata))
customerdata


# In[3]:


customerdata.columns


# ## Data Cleaning

# In[4]:


customerdata.info()


# ##### From the above output, we can conclude and note that:
# 
# ##### There are missing values in income
# ##### Dt_Customer that indicates the date a customer joined the database is not parsed as DateTime
# ##### There are some categorical features in our data frame; as there are some features in dtype: object). So we will need to encode them into numeric forms later.

# In[5]:


# Replace missing values of income feild with Income Mean Values
# Calculate the mean of the 'Income' column
income_mean = customerdata['Income'].mean().round(decimals=0)
income_mean


# In[6]:


# Replace missing values with the mean
customerdata['Income'].fillna(income_mean, inplace=True)


# ##### Create a feature out of "Dt_Customer" that indicates the number of days a customer is registered in the firm's database. However, in order to keep it simple, taking this value relative to the most recent customer in the record. Thus to get the values I must check the newest and oldest recorded dates.

# In[7]:


# Customer's time being enrolled
customerdata['Dt_Customer'] = pd.to_datetime(customerdata.Dt_Customer)
customerdata['Date_Collected'] = '01-01-2015'
customerdata['Date_Collected'] = pd.to_datetime(customerdata.Date_Collected)
customerdata['Time_Enrolled_Days'] = (customerdata['Date_Collected'] - customerdata['Dt_Customer']).dt.days


# In[8]:


print("Total categories in the feature Marital_Status:\n", customerdata["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", customerdata["Education"].value_counts())


# ##### Extract the "Age" of a customer by the "Year_Birth" indicating the birth year of the respective person.
# ##### Create another feature "Spent" indicating the total amount spent by the customer in various categories over the span of two years.
# ##### Create another feature "Living_With" out of "Marital_Status" to extract the living situation of couples.
# ##### Create a feature "Children" to indicate total children in a household that is, kids and teenagers.
# ##### To get further clarity of household, Creating feature indicating "Family_Size"
# ##### Create a feature "Is_Parent" to indicate parenthood status
# ##### Lastly, I will create three categories in the "Education" by simplifying its value counts.
# ##### Dropping some of the redundant features

# # Feature Engineering

# In[9]:


#Age of customer today 
customerdata["Age"] = 2024-customerdata["Year_Birth"]

#Total spendings on various items
customerdata["Spent"] = customerdata["MntWines"]+ customerdata["MntFruits"]+ customerdata["MntMeatProducts"]+ customerdata["MntFishProducts"]+ customerdata["MntSweetProducts"]+ customerdata["MntGoldProds"]

#Deriving living situation by marital status"Alone"
customerdata["Living_With"]=customerdata["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
customerdata["Children"]=customerdata["Kidhome"]+customerdata["Teenhome"]

#Feature for total members in the householde
customerdata["Family_Size"] = customerdata["Living_With"].replace({"Alone": 1, "Partner":2})+ customerdata["Children"]

#Feature pertaining parenthood
customerdata["Is_Parent"] = np.where(customerdata.Children> 0, 1, 0)

#Segmenting education levels in three groups
customerdata["Education"]=customerdata["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
customerdata=customerdata.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})


# In[10]:


#Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID","Date_Collected"]
customerdata = customerdata.drop(to_drop, axis=1)


# In[11]:


# Descriptive Statistics
customerdata.describe()


# ##### The above stats show some discrepancies in mean Income and Age and max Income and age.
# 
# ##### Do note that max-age is 131 years, As I calculated the age that would be today (i.e. 2024) and the data is old.
# 
# ##### It clearly says there are outliers in Income and age, taking only those data points which falls under 3 standard deviation.

# In[12]:


plt.figure(figsize=(10,6))
sns.set(style='whitegrid')
ax = sns.histplot(data=customerdata, x='Income', binwidth=10000, kde=True)
ax.set_title('Income chart of customers')


# In[13]:


plt.figure(figsize=(11,14), facecolor='lightyellow')
customerdata['Age'].value_counts().sort_index(ascending=False).plot(kind='barh')
plt.title('Age')
plt.show()


# In[14]:


#Plotting following features
To_Plot = [ "Income", "Recency", "Time_Enrolled_Days", "Age", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(customerdata[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()


# ##### There are a few outliers in the Income and Age features.Deleting the outliers in the customer data.

# In[15]:


# Removing outliers in income
from scipy import stats
# Remove observation with more than 3 in Z score
customerdata = customerdata[(np.abs(stats.zscore(customerdata['Income'])) < 3)]     

# Reset index
customerdata.reset_index(inplace=True)                          

customerdata = customerdata.drop(columns=['index'])


# In[16]:


customerdata.shape


# In[17]:

# Convert all categorical variables to numeric using one-hot encoding
customerdata_encoded = pd.get_dummies(customerdata)

# Compute the correlation matrix
corr_matrix = customerdata_encoded.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
#sns.heatmap(customerdata.corr())


# # Data preprocessing

# In[18]:


#Get list of categorical variables
s = (customerdata.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[19]:


#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    customerdata[i]=customerdata[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")


# In[20]:


customerdata.head()


# In[21]:


data_final = customerdata.copy()

#Scaling
scaler = StandardScaler()
scaler.fit(data_final)
scaled_ds = pd.DataFrame(scaler.transform(data_final),columns= data_final.columns )
X_scaled = scaler.fit_transform(data_final)
print("All features are now scaled")


# In[22]:


scaled_ds.head()


# # PCA

# In[23]:


pca = PCA(n_components=3)

# Fit PCA to the scaled data
pca.fit(X_scaled)

# Transform the data to its principal components
X_pca = pca.transform(X_scaled)


# In[24]:


import matplotlib.pyplot as plt

# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()


# # Perform Clustering

# ### Elbow Method to determine the number of clusters to be formed
# ### Model evaluation using Silhouette score

# In[25]:


from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
 kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
 kmeans.fit(X_pca)
 wcss.append(kmeans.inertia_)
plt.figure(figsize=(16,8))
plt.plot(range(1,11),wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# #### From above graph we can choose number of clusters as 4. Lets see for silhouette score for more details.

# In[26]:


Elbow_M = KElbowVisualizer(KMeans(), k=(3,10), metric='silhouette')
Elbow_M.fit(X_pca)
Elbow_M.show();


# In[27]:


from sklearn.metrics import silhouette_score
min_clusters = 2
max_clusters = 10
cluster_range = range(min_clusters, max_clusters + 1)

# Initialize lists to store silhouette scores
silhouette_scores = []

# Calculate Silhouette Score for each number of clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_ds)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_pca, labels)
    silhouette_scores.append(silhouette_avg)
    print("The cluster:", k , "average silhouette score is:", silhouette_avg)
# Plot Silhouette Scores versus number of clusters
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xticks(cluster_range)
plt.grid(True)

# Find the index of the maximum silhouette score
max_index = np.argmax(silhouette_scores)

# Annotate the elbow point
plt.annotate('Elbow Point', xy=(cluster_range[max_index], silhouette_scores[max_index]),
             xytext=(cluster_range[max_index] - 1, silhouette_scores[max_index] + 0.02),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.show()


# In[28]:


silhouette_scores = []
for i in range(2,10):
    m1=KMeans(n_clusters=i, random_state=42)
    c = m1.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, m1.fit_predict(X_pca))) 
plt.bar(range(2,10), silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 20) 
plt.ylabel('S(i)', fontsize = 20) 
plt.show()


# #### Although score of 3_clusters to 5_clusters is decent  silhouette_score however we are choosing 3_clusters because observations seems to be more evenly distributed among the clusters and making stratgey for 3 cluster is much more convinient and silhouette scores are droping after cluster 3 Scores 

# # K-Means Clsutering
# 
# #### KMeans is a popular unsupervised learning algorithm used for clustering. The algorithm works by dividing a set of observations into a predetermined number of clusters. The number of clusters is determined by the user before the algorithm is run. KMeans clustering works by first randomly initializing a set of centroids for each cluster. The centroids are points that represent the center of each cluster. The algorithm then iteratively assigns each observation to its closest centroid and updates the centroid position based on the new assignments. The algorithm repeats this process until the centroids no longer move or a maximum number of iterations is reached.

# In[29]:


# Training a predicting using K-Means Algorithm.

kmeans=KMeans(n_clusters=3, random_state=42).fit(X_pca)
pred=kmeans.predict(X_pca)


# Appending those cluster value into main dataframe (without standard-scalar)

customerdata['K_Means_cluster'] = pred + 1


# In[30]:


# Analyze the characteristics of each cluster
K_cluster_means = customerdata.groupby('K_Means_cluster').mean()
print(K_cluster_means)


# In[31]:


pl = sns.countplot(x=customerdata["K_Means_cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()


# In[32]:


pl = sns.scatterplot(data = customerdata,x=customerdata["Spent"], y=customerdata["Income"],hue=customerdata["K_Means_cluster"])
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# ##### group 1: low spending & low income
# ##### group 2: high spending & average income
# ##### group 3: high spending & high income

# In[33]:


# Plot the data points colored by predicted cluster labels
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=customerdata['K_Means_cluster'], cmap='viridis',
            marker='o', alpha=0.5)

# Plot cluster centroids (if available)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], c='red',
            marker='x', s=100, label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[34]:


plt.figure()
pl=sns.swarmplot(x=customerdata["K_Means_cluster"], y=customerdata["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=customerdata["K_Means_cluster"], y=customerdata["Spent"])
plt.show()


# In[35]:


plt.figure()
pl=sns.swarmplot(x=customerdata["K_Means_cluster"], y=customerdata["Education"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=customerdata["K_Means_cluster"], y=customerdata["Education"])
plt.show()


# ##### Next, Will be looking at the detailed distribution of clusters as per the various products in the data. Namely: Wines, Fruits, Meat, Fish, Sweets and Gold

# In[36]:


#Creating a feature to get a sum of accepted promotions 
customerdata["Total_Promos"] = customerdata["AcceptedCmp1"]+ customerdata["AcceptedCmp2"]+ customerdata["AcceptedCmp3"]+ customerdata["AcceptedCmp4"]+ customerdata["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=customerdata["Total_Promos"],hue=customerdata["K_Means_cluster"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()


# ##### There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one take part in all 5 of them. Perhaps better-targeted and well-planned campaigns are required to boost sales.

# In[37]:


#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=customerdata["NumDealsPurchases"],x=customerdata["K_Means_cluster"])
pl.set_title("Number of Deals Purchased")
plt.show()


# ## Unlike campaigns, the deals offered did well. It has best outcome with cluster 2 and cluster 3. However, our star customers cluster 1 are not much into the deals.

# ### Profiling the clusters

# ##### Now that we have formed the clusters and looked at their purchasing habits. Let us see who all are there in these clusters. For that, we will be profiling the clusters formed and come to a conclusion about who is our star customer and who needs more attention from the retail store's marketing team.
# 
# ##### To decide that I will be plotting some of the features that are indicative of the customer's personal traits in light of the cluster they are in. On the basis of the outcomes, I will be arriving at the conclusions.

# In[38]:


Personal = [ "Kidhome","Teenhome","Time_Enrolled_Days", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=customerdata[i], y=customerdata["Spent"], hue =customerdata["K_Means_cluster"], kind="kde")
    plt.show()


# ##### Cluster 1:
# ##### Most are education background graduates
# ##### Age range between = 20-65
# ##### Income is about 20k to 60k
# ##### Most have 1-2 kids and some no kids
# ##### Relationship status wise majority are 'Taken' and some 'Single'
# ##### Spending is very low: less than 200 bucks
# 
# ##### Cluster 2:
# ##### Majority are graduated, some have done masters and phd
# ##### Age range between = 22-68
# ##### Income is about 60k to 100k
# ##### No kids with maybe a few outliers
# ##### Relationship status wise majority are 'Taken' and some 'Single'
# ##### 750 < spending > 2500
# 
# ##### Cluster 3:
# ##### Majority are graduated and phd, some have done masters
# ##### Age range Between = 35-65
# ##### Income is about 400k to 500k
# ##### Majority have only 1 kid and some have 2 kids
# ##### Majority are 'Taken', very few 'Single'
# ##### 250 < spending > 1500

# # Model Building - K-Means Clusters

# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[40]:


customerdata


# In[41]:


X = customerdata.drop(columns=["K_Means_cluster"])
y = customerdata['K_Means_cluster']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# Initialize all classifiers with the same set of hyperparameters
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name, clf in classifiers.items():
    print(f"\033[1mTraining {name}...\033[0m")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy:.2f}")
    print(f"Classification Report of {name}:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    if len(np.unique(y_test)) == 2:  # Binary classification
        if hasattr(clf, "decision_function"):
            y_decision = clf.decision_function(X_test)
        else:
            y_decision = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_decision)
        auc = roc_auc_score(y_test, y_decision)
        print("AUC:", auc)
    else:  # Multiclass classification
        y_proba = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        print("AUC:", auc)
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())
    print("--------------------------------------------------")


# Based on the performance metrics provided for each classifier, we can summarize the best fit for the K-means clustering approach by considering accuracy, AUC, and cross-validation scores. Here's a detailed summary of the results for the various classifiers:
# 
# ### Summary of Classifier Performance
# 
# 1. **Logistic Regression**
#    - **Accuracy**: 0.86
#    - **AUC**: 0.9614
#    - **Mean Cross-Validation Score**: 0.8602
#    - **Classification Report**:
#      - Precision: High for class 1 (0.96), moderate for classes 2 and 3 (0.77, 0.77)
#      - Recall: Very high for class 1 (0.95), moderate for classes 2 and 3 (0.74, 0.80)
# 
# 2. **Support Vector Machine (SVM)**
#    - **Accuracy**: 0.79
#    - **AUC**: 0.9099
#    - **Mean Cross-Validation Score**: 0.7836
#    - **Classification Report**:
#      - Precision: High for class 1 (0.90), moderate for classes 2 and 3 (0.75, 0.66)
#      - Recall: Very high for class 1 (0.91), moderate for classes 2 and 3 (0.60, 0.73)
# 
# 3. **Random Forest**
#    - **Accuracy**: 0.99
#    - **AUC**: 0.9996
#    - **Mean Cross-Validation Score**: 0.9807
#    - **Classification Report**:
#      - Precision: Very high across all classes (0.98, 0.99, 0.99)
#      - Recall: Very high across all classes (0.99, 0.99, 0.98)
# 
# 4. **Gradient Boosting**
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9997
#    - **Mean Cross-Validation Score**: 0.9816
#    - **Classification Report**:
#      - Precision: Very high across all classes (0.99, 0.98, 0.98)
#      - Recall: Very high across all classes (0.99, 0.98, 0.98)
# 
# 5. **K-Nearest Neighbors (KNN)**
#    - **Accuracy**: 0.86
#    - **AUC**: 0.9385
#    - **Mean Cross-Validation Score**: 0.8522
#    - **Classification Report**:
#      - Precision: High for class 1 (0.94), moderate for classes 2 and 3 (0.78, 0.77)
#      - Recall: Very high for class 1 (0.97), moderate for classes 2 and 3 (0.72, 0.77)
# 
# 6. **Naive Bayes**
#    - **Accuracy**: 0.96
#    - **AUC**: 0.9938
#    - **Mean Cross-Validation Score**: 0.9574
#    - **Classification Report**:
#      - Precision: Very high for class 1 (1.00), high for classes 2 and 3 (0.98, 0.88)
#      - Recall: Very high for classes 1 and 3 (0.95, 0.98), high for class 2 (0.95)
# 
# 7. **Decision Tree**
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9848
#    - **Mean Cross-Validation Score**: 0.9651
#    - **Classification Report**:
#      - Precision: Very high for classes 1 and 2 (0.99, 0.99), high for class 3 (0.96)
#      - Recall: Very high for classes 1 and 3 (0.98, 0.98), high for class 2 (0.98)
# 
# ### Best Fit for K-Means
# 
# When considering the best fit for K-means clustering, we focus on models that offer a balance of high accuracy, high AUC, and consistent performance across cross-validation.
# 
# 1. **Random Forest**:
#    - **Accuracy**: 0.99 (highest among all models)
#    - **AUC**: 0.9996 (highest)
#    - **Mean Cross-Validation Score**: 0.9807 (among the highest)
#    - **Summary**: Random Forest demonstrates the highest accuracy, very high AUC, and very high mean cross-validation score, making it the most robust and reliable model overall.
# 
# 2. **Gradient Boosting**:
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9997 (highest)
#    - **Mean Cross-Validation Score**: 0.9816 (highest)
#    - **Summary**: Gradient Boosting also performs exceptionally well with very high accuracy and AUC, making it another strong candidate.
# 
# 3. **Decision Tree**:
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9848
#    - **Mean Cross-Validation Score**: 0.9651
#    - **Summary**: Decision Tree performs very well with high accuracy and AUC, and a strong cross-validation score, making it a solid choice.
# 
# 4. **Naive Bayes**:
#    - **Accuracy**: 0.96
#    - **AUC**: 0.9938
#    - **Mean Cross-Validation Score**: 0.9574
#    - **Summary**: Naive Bayes performs very well with high accuracy and AUC, and a strong cross-validation score, making it a solid choice.
# 
# ### Conclusion
# 
# The **Random Forest** classifier is the best fit for K-means clustering due to its highest accuracy, very high AUC, and high mean cross-validation score, demonstrating robustness and reliability across different data splits. **Gradient Boosting** is a very close second, with excellent performance metrics making it another highly suitable option. **Decision Tree** and **Naive Bayes** are also strong performers and can be considered depending on specific needs and constraints of the clustering task.

# # Agglomerative Clustering
# 
# ##### Agglomerative Clustering is another popular clustering algorithm that works by iteratively merging the closest pairs of clusters until all the observations belong to a single cluster. The algorithm starts by assigning each observation to its own cluster. It then iteratively merges the closest pair of clusters based on a distance metric until all observations belong to a single cluster.

# In[42]:


# Perform hierarchical clustering
agglomerative_cluster = AgglomerativeClustering(n_clusters=3)
clusters = agglomerative_cluster.fit_predict(X_pca)
# Add cluster labels to the original dataset
customerdata['Agglomerative_Cluster'] = clusters


# In[43]:


# Analyze the characteristics of each cluster
Ag_cluster_means = customerdata.groupby('Agglomerative_Cluster').mean()
print(Ag_cluster_means)


# In[44]:


pl = sns.scatterplot(data = customerdata,x=customerdata["Spent"], y=customerdata["Income"],hue=customerdata["Agglomerative_Cluster"])
pl.set_title("Agglomerative Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# In[45]:


pl = sns.countplot(x=customerdata["Agglomerative_Cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()


# In[46]:


plt.figure()
pl=sns.swarmplot(x=customerdata["Agglomerative_Cluster"], y=customerdata["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=customerdata["Agglomerative_Cluster"], y=customerdata["Spent"])
plt.show()


# In[47]:


#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=customerdata["NumDealsPurchases"],x=customerdata["Agglomerative_Cluster"])
pl.set_title("Number of Deals Purchased")
plt.show()


# In[48]:


#for more details on the purchasing style 
Places =["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",  "NumWebVisitsMonth"] 

for i in Places:
    plt.figure()
    sns.jointplot(x=customerdata[i],y = customerdata["Spent"],hue=customerdata["Agglomerative_Cluster"])
    plt.show()


# ### Profiling Clustering

# In[49]:


Personal = [ "Kidhome","Teenhome","Time_Enrolled_Days", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=customerdata[i], y=customerdata["Spent"], hue =customerdata["Agglomerative_Cluster"], kind="kde")
    plt.show()


# # Model Building - Agglomerative Cluster

# In[50]:


X = customerdata.drop(columns=["Agglomerative_Cluster"])
y = customerdata['Agglomerative_Cluster']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# Initialize all classifiers with the same set of hyperparameters
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name, clf in classifiers.items():
    print(f"\033[1mTraining {name}...\033[0m")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy:.2f}")
    print(f"Classification Report of {name}:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    if len(np.unique(y_test)) == 2:  # Binary classification
        if hasattr(clf, "decision_function"):
            y_decision = clf.decision_function(X_test)
        else:
            y_decision = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_decision)
        auc = roc_auc_score(y_test, y_decision)
        print("AUC:", auc)
    else:  # Multiclass classification
        y_proba = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        print("AUC:", auc)
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())
    print("--------------------------------------------------")


# Based on the performance metrics provided for each classifier, we can summarize the best fit in an augmented cluster approach by considering accuracy, AUC, and cross-validation scores. Here’s a detailed summary of the results:
# 
# ### Summary of Classifier Performance
# 
# 1. **Logistic Regression**
#    - **Accuracy**: 0.87
#    - **AUC**: 0.9585
#    - **Mean Cross-Validation Score**: 0.8459
#    - **Classification Report**:
#      - Precision: High for class 0 (0.95), moderate for classes 1 and 2 (0.77, 0.76)
#      - Recall: Very high for class 0 (0.98), moderate for classes 1 and 2 (0.78, 0.69)
# 
# 2. **Support Vector Machine (SVM)**
#    - **Accuracy**: 0.76
#    - **AUC**: 0.8996
#    - **Mean Cross-Validation Score**: 0.7845
#    - **Classification Report**:
#      - Precision: High for class 0 (0.85), moderate for classes 1 and 2 (0.70, 0.58)
#      - Recall: Very high for class 0 (0.93), moderate for classes 1 and 2 (0.59, 0.53)
# 
# 3. **Random Forest**
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9996
#    - **Mean Cross-Validation Score**: 0.9727
#    - **Classification Report**:
#      - Precision: Very high across all classes (0.99, 1.00, 0.95)
#      - Recall: Very high across all classes (0.99, 0.98, 0.97)
# 
# 4. **Gradient Boosting**
#    - **Accuracy**: 0.99
#    - **AUC**: 0.9994
#    - **Mean Cross-Validation Score**: 0.9673
#    - **Classification Report**:
#      - Precision: Very high across all classes (0.99, 1.00, 0.97)
#      - Recall: Very high across all classes (0.99, 0.99, 0.97)
# 
# 5. **K-Nearest Neighbors (KNN)**
#    - **Accuracy**: 0.82
#    - **AUC**: 0.9204
#    - **Mean Cross-Validation Score**: 0.8315
#    - **Classification Report**:
#      - Precision: High for class 0 (0.91), moderate for classes 1 and 2 (0.74, 0.67)
#      - Recall: High for class 0 (0.94), moderate for classes 1 and 2 (0.71, 0.63)
# 
# 6. **Naive Bayes**
#    - **Accuracy**: 0.94
#    - **AUC**: 0.9871
#    - **Mean Cross-Validation Score**: 0.9279
#    - **Classification Report**:
#      - Precision: Very high for class 0 (1.00), high for classes 1 and 2 (0.91, 0.83)
#      - Recall: Very high for class 0 (0.93), high for classes 1 and 2 (0.91, 0.96)
# 
# 7. **Decision Tree**
#    - **Accuracy**: 0.96
#    - **AUC**: 0.9722
#    - **Mean Cross-Validation Score**: 0.9534
#    - **Classification Report**:
#      - Precision: Very high for classes 0 and 1 (0.98, 0.99), high for class 2 (0.90)
#      - Recall: Very high for classes 0 and 1 (0.97, 0.97), high for class 2 (0.95)
# 
# ### Best Fit in Augmented Cluster
# 
# When considering the best fit for an augmented cluster approach, we should focus on models that offer a balance of high accuracy, high AUC, and consistent performance across cross-validation.
# 
# 1. **Gradient Boosting**:
#    - **Accuracy**: 0.99 (highest among all models)
#    - **AUC**: 0.9994 (very close to the highest)
#    - **Mean Cross-Validation Score**: 0.9673 (among the highest)
#    - **Summary**: Gradient Boosting demonstrates the highest accuracy, very high AUC, and very high mean cross-validation score, making it the most robust and reliable model overall.
# 
# 2. **Random Forest**:
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9996 (highest among all models)
#    - **Mean Cross-Validation Score**: 0.9727 (highest)
#    - **Summary**: Random Forest also performs exceptionally well with the highest AUC and very high accuracy and cross-validation score, making it another strong candidate.
# 
# 3. **Decision Tree**:
#    - **Accuracy**: 0.96
#    - **AUC**: 0.9722
#    - **Mean Cross-Validation Score**: 0.9534
#    - **Summary**: Decision Tree performs very well with high accuracy and AUC, and a strong cross-validation score, making it a solid choice for certain scenarios.
# 
# ### Conclusion
# 
# The **Gradient Boosting** classifier is the best fit in an augmented cluster approach due to its highest accuracy, very high AUC, and high mean cross-validation score, demonstrating robustness and reliability across different data splits. **Random Forest** is a very close second, with excellent performance metrics making it another highly suitable option. **Decision Tree** is also a strong performer and can be considered depending on specific needs and constraints of the clustering task.

# # Hybrid clustering

# In[52]:


from sklearn.cluster import DBSCAN
# Parameters for hybrid clustering
k_kmeans = 4  # Number of clusters for KMeans
eps_dbscan = 0.5  # DBSCAN epsilon parameter
min_samples_dbscan = 4  # DBSCAN min_samples parameter

# Perform hybrid clustering
# KMeans clustering
kmeans = KMeans(n_clusters=k_kmeans)
kmeans.fit(scaled_ds)
kmeans_labels = kmeans.labels_
    
# DBSCAN clustering
dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
dbscan.fit(scaled_ds)
dbscan_labels = dbscan.labels_
    
# Combine cluster labels from KMeans and DBSCAN
hybrid_labels = np.where(dbscan_labels == -1, kmeans_labels, dbscan_labels)
    
# Add the hybrid cluster labels to your DataFrame
customerdata['Hybrid_Cluster'] = hybrid_labels

# Print size of each hybrid cluster
print(customerdata['Hybrid_Cluster'].value_counts())

silhouette_avg = silhouette_score(scaled_ds, hybrid_labels)
print("The average silhouette score for the hybrid clustering model is:", silhouette_avg)


# In[53]:


# Plotting
plt.figure(figsize=(8, 6))
plt.hist(customerdata['Hybrid_Cluster'], bins='auto', color='skyblue', edgecolor='black')
plt.xlabel('HYbrid Cluster')
plt.ylabel('Frequency')
plt.title('Distribution of Hybrid Clusters')
plt.grid(True)
plt.show()


# In[54]:


# Parameters for hybrid clustering
k_kmeans = 3  # Number of clusters for KMeans
eps_dbscan = 0.5  # DBSCAN epsilon parameter
min_samples_dbscan = 4  # DBSCAN min_samples parameter

# Perform hybrid clustering
# KMeans clustering
kmeans = KMeans(n_clusters=k_kmeans)
kmeans.fit(scaled_ds)
kmeans_labels = kmeans.labels_
    
# DBSCAN clustering
dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
dbscan.fit(scaled_ds)
dbscan_labels = dbscan.labels_
    
# Combine cluster labels from KMeans and DBSCAN
hybrid_labels = np.where(dbscan_labels == -1, kmeans_labels, dbscan_labels)
    
# Add the hybrid cluster labels to your DataFrame
customerdata['Hybrid_Cluster'] = hybrid_labels

# Print size of each hybrid cluster
print(customerdata['Hybrid_Cluster'].value_counts())

silhouette_avg = silhouette_score(scaled_ds, hybrid_labels)
print("The average silhouette score for the hybrid clustering model is:", silhouette_avg)


# In[55]:


# Plotting
plt.figure(figsize=(8, 6))
plt.hist(customerdata['Hybrid_Cluster'], bins='auto', color='skyblue', edgecolor='black')
plt.xlabel('HYbrid Cluster')
plt.ylabel('Frequency')
plt.title('Distribution of Hybrid Clusters')
plt.grid(True)
plt.show()


# ## Profiling Clusters

# In[56]:


Personal = [ "Kidhome","Teenhome","Time_Enrolled_Days", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=customerdata[i], y=customerdata["Spent"], hue =customerdata["Hybrid_Cluster"], kind="kde")
    plt.show()


# # Model Building - Hybrid Clustering

# In[57]:


X = customerdata.drop(columns=["Hybrid_Cluster"])
y = customerdata['Hybrid_Cluster']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

# Initialize all classifiers with the same set of hyperparameters
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name, clf in classifiers.items():
    print(f"\033[1mTraining {name}...\033[0m")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy:.2f}")
    print(f"Classification Report of {name}:")
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    if len(np.unique(y_test)) == 2:  # Binary classification
        if hasattr(clf, "decision_function"):
            y_decision = clf.decision_function(X_test)
        else:
            y_decision = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_decision)
        auc = roc_auc_score(y_test, y_decision)
        print("AUC:", auc)
    else:  # Multiclass classification
        y_proba = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        
        print("AUC:", auc)
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())
    print("--------------------------------------------------")


# Based on the performance metrics provided for each classifier, we can identify the best fit in a hybrid cluster approach by considering accuracy, AUC, and cross-validation scores. Here’s a summary of the results:
# 
# ### Summary of Classifier Performance
# 
# 1. **Logistic Regression**
#    - Accuracy: 0.87
#    - AUC: 0.9648
#    - Mean Cross-Validation Score: 0.8611
# 
# 2. **Support Vector Machine (SVM)**
#    - Accuracy: 0.79
#    - AUC: 0.9075
#    - Mean Cross-Validation Score: 0.7814
# 
# 3. **Random Forest**
#    - Accuracy: 0.98
#    - AUC: 0.9995
#    - Mean Cross-Validation Score: 0.9848
# 
# 4. **Gradient Boosting**
#    - Accuracy: 0.99
#    - AUC: 0.9989
#    - Mean Cross-Validation Score: 0.9852
# 
# 5. **K-Nearest Neighbors (KNN)**
#    - Accuracy: 0.85
#    - AUC: 0.9356
#    - Mean Cross-Validation Score: 0.8580
# 
# 6. **Naive Bayes**
#    - Accuracy: 0.95
#    - AUC: 0.9958
#    - Mean Cross-Validation Score: 0.9552
# 
# 7. **Decision Tree**
#    - Accuracy: 0.98
#    - AUC: 0.9817
#    - Mean Cross-Validation Score: 0.9785
# 
# ### Best Fit in Hybrid Cluster
# 
# When considering the best fit for a hybrid cluster, we should look for models that offer a balance of high accuracy, high AUC, and consistent performance across cross-validation.
# 
# 1. **Gradient Boosting**:
#    - **Accuracy**: 0.99 (highest among all models)
#    - **AUC**: 0.9989 (very close to the highest)
#    - **Mean Cross-Validation Score**: 0.9852 (highest among all models)
#    - **Summary**: Gradient Boosting shows the highest accuracy and very high AUC, along with the highest mean cross-validation score, indicating it is the most robust model overall.
# 
# 2. **Random Forest**:
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9995 (highest among all models)
#    - **Mean Cross-Validation Score**: 0.9848 (very close to the highest)
#    - **Summary**: Random Forest also performs exceptionally well with the highest AUC and nearly the highest accuracy and cross-validation score, making it another strong candidate.
# 
# 3. **Decision Tree**:
#    - **Accuracy**: 0.98
#    - **AUC**: 0.9817
#    - **Mean Cross-Validation Score**: 0.9785
#    - **Summary**: Decision Tree performs similarly to Random Forest but with slightly lower AUC and cross-validation score.
# 
# ### Conclusion
# 
# The **Gradient Boosting** classifier is the best fit in a hybrid cluster approach due to its highest accuracy, very high AUC, and the highest mean cross-validation score, demonstrating robustness and reliability across different data splits. **Random Forest** is a very close second, with excellent performance metrics making it another suitable option. These models should be preferred for their overall effectiveness in providing high-quality clustering and classification performance.

# # Hence we will deploy model using Gradient Boosting classifier as two clusters giving 0.99 accuracy .

# In[ ]:




