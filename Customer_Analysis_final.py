#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
customerdata = pd.read_excel("../DS Project1/marketing_campaign.xlsx")
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


sns.heatmap(customerdata.corr())


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


Elbow_M = KElbowVisualizer(KMeans(), k=(2,10), metric='silhouette')
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


# ##### Next, Will be looking at the detailed distribution of clusters as per the various products in the data. Namely: Wines, Fruits, Meat, Fish, Sweets and Gold

# In[35]:


#Creating a feature to get a sum of accepted promotions 
customerdata["Total_Promos"] = customerdata["AcceptedCmp1"]+ customerdata["AcceptedCmp2"]+ customerdata["AcceptedCmp3"]+ customerdata["AcceptedCmp4"]+ customerdata["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=customerdata["Total_Promos"],hue=customerdata["K_Means_cluster"])
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()


# ##### There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one take part in all 5 of them. Perhaps better-targeted and well-planned campaigns are required to boost sales.

# In[36]:


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

# In[37]:


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

# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[39]:


def plot_precision_recall_curve(y_test, y_decision, model_name):
    if len(np.unique(y_test)) == 2:  # Binary classification
        precision, recall, _ = precision_recall_curve(y_test, y_decision)
    else:  # Multiclass classification
        precision = dict()
        recall = dict()
        for i in range(len(np.unique(y_test))):
            precision[i], recall[i], _ = precision_recall_curve(y_test, y_decision[:, i], pos_label=i)

    plt.figure(figsize=(8, 6))
    if len(np.unique(y_test)) == 2:  # Binary classification
        plt.plot(recall, precision, marker='.')
    else:
        for i in range(len(np.unique(y_test))):
            plt.plot(recall[i], precision[i], marker='.', label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.show()


# In[40]:


def plot_roc_curve(y_test, y_decision, model_name):
    if len(np.unique(y_test)) == 2:  # Binary classification
        fpr, tpr, _ = roc_curve(y_test, y_decision)
        auc_score = roc_auc_score(y_test, y_decision)
    else:  # Multiclass classification
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_decision[:, i], pos_label=i)
            roc_auc[i] = roc_auc_score(y_test, y_decision, multi_class='ovr')

        auc_score = np.mean(list(roc_auc.values()))

    plt.figure(figsize=(8, 6))
    if len(np.unique(y_test)) == 2:  # Binary classification
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    else:
        for i in range(len(np.unique(y_test))):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.show()


# In[41]:


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[42]:


customerdata.columns


# In[43]:


k_means_data = customerdata.drop(columns=['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5','AcceptedCmp1', 'AcceptedCmp2','Complain', 'Response'])
k_means_data


# In[46]:


# this is for all models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score,precision_recall_curve
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


# In[47]:


X = k_means_data.drop(columns=["K_Means_cluster"])
y = k_means_data['K_Means_cluster']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Fit and evaluate models
for model_name, model in classifiers.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = model.score(X_test, y_test)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"\033[1mModel Name: {model_name}...\033[0m")
    print(f"Accuracy of {model_name}: {accuracy:.2f}")
    print(f"Classification Report of {model_name}:\n{class_report}")
    print(f"Confusion Matrix of {model_name}:\n{conf_matrix}")
    print(f"AUC of {model_name}: {auc_score:.2f}")
    print(f"Cross-Validation Scores of {model_name}: {cv_scores}")
    print(f"Mean Cross-Validation Score of {model_name}: {cv_scores.mean():.2f}")
    print("--------------------------------------------------")

    print(f"\033[1mEvaluating {model_name}...\033[0m")
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)
    plot_precision_recall_curve(y_test, y_proba, model_name)


# # Summary
# Based on the provided evaluation metrics, here's a detailed summary and comparison of the models:
# 
# #### 1. **Logistic Regression**
# - **Accuracy**: 0.86
# - **AUC**: 0.96
# - **Mean Cross-Validation Score**: 0.86
# - **Precision/Recall**:
#   - Class 1: High (Precision: 0.96, Recall: 0.95)
#   - Class 2: Moderate (Precision: 0.77, Recall: 0.76)
#   - Class 3: Moderate (Precision: 0.76, Recall: 0.79)
# - **Confusion Matrix**:
#   - Class 1: [211, 0, 10]
#   - Class 2: [1, 71, 22]
#   - Class 3: [7, 21, 104]
# 
# #### 2. **Support Vector Machine (SVM)**
# - **Accuracy**: 0.79
# - **AUC**: 0.91
# - **Mean Cross-Validation Score**: 0.79
# - **Precision/Recall**:
#   - Class 1: High (Precision: 0.90, Recall: 0.91)
#   - Class 2: Moderate (Precision: 0.75, Recall: 0.60)
#   - Class 3: Moderate (Precision: 0.66, Recall: 0.73)
# - **Confusion Matrix**:
#   - Class 1: [202, 0, 19]
#   - Class 2: [6, 56, 32]
#   - Class 3: [16, 19, 97]
# 
# #### 3. **Random Forest**
# - **Accuracy**: 0.98
# - **AUC**: 1.00
# - **Mean Cross-Validation Score**: 0.97
# - **Precision/Recall**:
#   - Class 1: High (Precision: 0.98, Recall: 0.98)
#   - Class 2: High (Precision: 0.98, Recall: 0.97)
#   - Class 3: High (Precision: 0.97, Recall: 0.97)
# - **Confusion Matrix**:
#   - Class 1: [217, 2, 2]
#   - Class 2: [1, 91, 2]
#   - Class 3: [4, 0, 128]
# 
# #### 4. **Gradient Boosting**
# - **Accuracy**: 0.98
# - **AUC**: 1.00
# - **Mean Cross-Validation Score**: 0.97
# - **Precision/Recall**:
#   - Class 1: High (Precision: 0.99, Recall: 0.98)
#   - Class 2: High (Precision: 0.99, Recall: 0.99)
#   - Class 3: High (Precision: 0.97, Recall: 0.98)
# - **Confusion Matrix**:
#   - Class 1: [217, 1, 3]
#   - Class 2: [0, 93, 1]
#   - Class 3: [2, 0, 130]
# 
# #### 5. **K-Nearest Neighbors (KNN)**
# - **Accuracy**: 0.86
# - **AUC**: 0.94
# - **Mean Cross-Validation Score**: 0.85
# - **Precision/Recall**:
#   - Class 1: High (Precision: 0.94, Recall: 0.97)
#   - Class 2: Moderate (Precision: 0.78, Recall: 0.72)
#   - Class 3: Moderate (Precision: 0.77, Recall: 0.77)
# - **Confusion Matrix**:
#   - Class 1: [214, 0, 7]
#   - Class 2: [3, 68, 23]
#   - Class 3: [11, 19, 102]
# 
# #### 6. **Naive Bayes**
# - **Accuracy**: 0.94
# - **AUC**: 0.99
# - **Mean Cross-Validation Score**: 0.94
# - **Precision/Recall**:
#   - Class 1: Very High (Precision: 1.00, Recall: 0.95)
#   - Class 2: High (Precision: 0.92, Recall: 0.93)
#   - Class 3: High (Precision: 0.87, Recall: 0.93)
# - **Confusion Matrix**:
#   - Class 1: [209, 0, 12]
#   - Class 2: [0, 87, 7]
#   - Class 3: [1, 8, 123]
# 
# #### 7. **Decision Tree**
# - **Accuracy**: 0.95
# - **AUC**: 0.96
# - **Mean Cross-Validation Score**: 0.96
# - **Precision/Recall**:
#   - Class 1: High (Precision: 0.98, Recall: 0.96)
#   - Class 2: High (Precision: 0.93, Recall: 0.96)
#   - Class 3: High (Precision: 0.93, Recall: 0.93)
# - **Confusion Matrix**:
#   - Class 1: [213, 3, 5]
#   - Class 2: [0, 90, 4]
#   - Class 3: [5, 4, 123]
# 
# ### Best-Fit Model Analysis:
# - **Accuracy**: Random Forest and Gradient Boosting both achieved the highest accuracy of 0.98.
# - **AUC**: Random Forest and Gradient Boosting both achieved a perfect AUC of 1.00.
# - **Cross-Validation**: Random Forest and Gradient Boosting both had a mean cross-validation score of 0.97, indicating consistent performance across different data splits.
# - **Precision and Recall**: Both Random Forest and Gradient Boosting demonstrated high precision and recall across all classes, indicating reliable performance in distinguishing between classes.
# 
# ### Conclusion:
# **Random Forest** and **Gradient Boosting** are the best-fit models based on the provided evaluation metrics. Both models have identical performance with the highest accuracy, perfect AUC, consistent cross-validation scores, and high precision and recall across all classes. These models are recommended for further use and deployment.

# # Agglomerative Clustering
# 
# ##### Agglomerative Clustering is another popular clustering algorithm that works by iteratively merging the closest pairs of clusters until all the observations belong to a single cluster. The algorithm starts by assigning each observation to its own cluster. It then iteratively merges the closest pair of clusters based on a distance metric until all observations belong to a single cluster.

# In[48]:


# Perform hierarchical clustering
agglomerative_cluster = AgglomerativeClustering(n_clusters=3)
clusters = agglomerative_cluster.fit_predict(X_pca)
# Add cluster labels to the original dataset
customerdata['Agglomerative_Cluster'] = clusters


# In[49]:


# Analyze the characteristics of each cluster
Ag_cluster_means = customerdata.groupby('Agglomerative_Cluster').mean()
print(Ag_cluster_means)


# In[50]:


pl = sns.scatterplot(data = customerdata,x=customerdata["Spent"], y=customerdata["Income"],hue=customerdata["Agglomerative_Cluster"])
pl.set_title("Agglomerative Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# In[51]:


pl = sns.countplot(x=customerdata["Agglomerative_Cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()


# In[52]:


plt.figure()
pl=sns.swarmplot(x=customerdata["Agglomerative_Cluster"], y=customerdata["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=customerdata["Agglomerative_Cluster"], y=customerdata["Spent"])
plt.show()


# In[53]:


#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=customerdata["NumDealsPurchases"],x=customerdata["Agglomerative_Cluster"])
pl.set_title("Number of Deals Purchased")
plt.show()


# In[54]:


#for more details on the purchasing style 
Places =["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",  "NumWebVisitsMonth"] 

for i in Places:
    plt.figure()
    sns.jointplot(x=customerdata[i],y = customerdata["Spent"],hue=customerdata["Agglomerative_Cluster"])
    plt.show()


# ### Profiling Clustering

# In[55]:


Personal = [ "Kidhome","Teenhome","Time_Enrolled_Days", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=customerdata[i], y=customerdata["Spent"], hue =customerdata["Agglomerative_Cluster"], kind="kde")
    plt.show()


# # Model Building - Agglomerative Cluster

# In[56]:


customerdata.columns


# In[57]:


Agglomerative_data = customerdata.drop(columns=['K_Means_cluster', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5','AcceptedCmp1', 'AcceptedCmp2','Complain', 'Response'])
Agglomerative_data


# In[58]:


X = Agglomerative_data.drop(columns=["Agglomerative_Cluster"])
y = Agglomerative_data['Agglomerative_Cluster']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Fit and evaluate models
for model_name, model in classifiers.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = model.score(X_test, y_test)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"\033[1mModel Name: {model_name}...\033[0m")
    print(f"Accuracy of {model_name}: {accuracy:.2f}")
    print(f"Classification Report of {model_name}:\n{class_report}")
    print(f"Confusion Matrix of {model_name}:\n{conf_matrix}")
    print(f"AUC of {model_name}: {auc_score:.2f}")
    print(f"Cross-Validation Scores of {model_name}: {cv_scores}")
    print(f"Mean Cross-Validation Score of {model_name}: {cv_scores.mean():.2f}")
    print("--------------------------------------------------")

    print(f"\033[1mEvaluating {model_name}...\033[0m")
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)
    plot_precision_recall_curve(y_test, y_proba, model_name)


# ### Summary of Model Evaluation:
# 
# #### 1. **Logistic Regression**
# - **Accuracy**: 0.85
# - **AUC**: 0.96
# - **Mean Cross-Validation Score**: 0.85
# - **Classification Report**:
#   - **Class 0**: Precision: 0.78, Recall: 0.74, F1-Score: 0.76
#   - **Class 1**: Precision: 0.95, Recall: 0.96, F1-Score: 0.95
#   - **Class 2**: Precision: 0.72, Recall: 0.73, F1-Score: 0.72
# - **Confusion Matrix**:
#   - [70, 1, 23]
#   - [0, 229, 10]
#   - [20, 11, 83]
# 
# #### 2. **Support Vector Machine (SVM)**
# - **Accuracy**: 0.77
# - **AUC**: 0.90
# - **Mean Cross-Validation Score**: 0.78
# - **Classification Report**:
#   - **Class 0**: Precision: 0.75, Recall: 0.61, F1-Score: 0.67
#   - **Class 1**: Precision: 0.85, Recall: 0.92, F1-Score: 0.88
#   - **Class 2**: Precision: 0.60, Recall: 0.60, F1-Score: 0.60
# - **Confusion Matrix**:
#   - [57, 11, 26]
#   - [0, 219, 20]
#   - [19, 27, 68]
# 
# #### 3. **Random Forest**
# - **Accuracy**: 0.96
# - **AUC**: 1.00
# - **Mean Cross-Validation Score**: 0.96
# - **Classification Report**:
#   - **Class 0**: Precision: 0.99, Recall: 0.97, F1-Score: 0.98
#   - **Class 1**: Precision: 0.97, Recall: 0.97, F1-Score: 0.97
#   - **Class 2**: Precision: 0.92, Recall: 0.95, F1-Score: 0.94
# - **Confusion Matrix**:
#   - [91, 0, 3]
#   - [1, 232, 6]
#   - [0, 6, 108]
# 
# #### 4. **Gradient Boosting**
# - **Accuracy**: 0.98
# - **AUC**: 1.00
# - **Mean Cross-Validation Score**: 0.97
# - **Classification Report**:
#   - **Class 0**: Precision: 0.99, Recall: 0.97, F1-Score: 0.98
#   - **Class 1**: Precision: 0.99, Recall: 0.97, F1-Score: 0.98
#   - **Class 2**: Precision: 0.94, Recall: 0.98, F1-Score: 0.96
# - **Confusion Matrix**:
#   - [91, 1, 2]
#   - [1, 233, 5]
#   - [0, 2, 112]
# 
# #### 5. **K-Nearest Neighbors (KNN)**
# - **Accuracy**: 0.84
# - **AUC**: 0.93
# - **Mean Cross-Validation Score**: 0.83
# - **Classification Report**:
#   - **Class 0**: Precision: 0.80, Recall: 0.70, F1-Score: 0.75
#   - **Class 1**: Precision: 0.92, Recall: 0.95, F1-Score: 0.93
#   - **Class 2**: Precision: 0.70, Recall: 0.71, F1-Score: 0.71
# - **Confusion Matrix**:
#   - [66, 5, 23]
#   - [0, 228, 11]
#   - [17, 16, 81]
# 
# #### 6. **Naive Bayes**
# - **Accuracy**: 0.92
# - **AUC**: 0.99
# - **Mean Cross-Validation Score**: 0.92
# - **Classification Report**:
#   - **Class 0**: Precision: 0.94, Recall: 0.93, F1-Score: 0.93
#   - **Class 1**: Precision: 0.98, Recall: 0.92, F1-Score: 0.95
#   - **Class 2**: Precision: 0.81, Recall: 0.92, F1-Score: 0.86
# - **Confusion Matrix**:
#   - [87, 0, 7]
#   - [1, 220, 18]
#   - [5, 4, 105]
# 
# #### 7. **Decision Tree**
# - **Accuracy**: 0.96
# - **AUC**: 0.96
# - **Mean Cross-Validation Score**: 0.95
# - **Classification Report**:
#   - **Class 0**: Precision: 0.97, Recall: 0.97, F1-Score: 0.97
#   - **Class 1**: Precision: 0.96, Recall: 0.97, F1-Score: 0.97
#   - **Class 2**: Precision: 0.94, Recall: 0.91, F1-Score: 0.92
# - **Confusion Matrix**:
#   - [91, 0, 3]
#   - [2, 233, 4]
#   - [1, 9, 104]
# 
# ### Best-Fit Model Analysis:
# 
# - **Accuracy**: Gradient Boosting (0.98) and Random Forest (0.96) have the highest accuracy.
# - **AUC**: Gradient Boosting and Random Forest both have a perfect AUC of 1.00.
# - **Cross-Validation Scores**: Gradient Boosting (0.97) and Random Forest (0.96) have the highest mean cross-validation scores.
# - **Precision, Recall, F1-Score**: Gradient Boosting and Random Forest both show high precision, recall, and F1-scores across all classes, with Gradient Boosting having a slight edge in precision and recall for Class 2.
# 
# ### Conclusion:
# **Gradient Boosting** is the best-fit model based on the provided evaluation metrics. It has the highest accuracy, perfect AUC, the highest mean cross-validation score, and high precision and recall across all classes. This model is recommended for further use and deployment.

# # Hybrid clustering

# In[60]:


from sklearn.cluster import DBSCAN
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


# In[61]:


# Plotting
plt.figure(figsize=(8, 6))
plt.hist(customerdata['Hybrid_Cluster'], bins='auto', color='skyblue', edgecolor='black')
plt.xlabel('HYbrid Cluster')
plt.ylabel('Frequency')
plt.title('Distribution of Hybrid Clusters')
plt.grid(True)
plt.show()


# ## Profiling Clusters

# In[62]:


Personal = [ "Kidhome","Teenhome","Time_Enrolled_Days", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=customerdata[i], y=customerdata["Spent"], hue =customerdata["Hybrid_Cluster"], kind="kde")
    plt.show()


# # Model Building - Hybrid Clustering

# In[63]:


customerdata.columns


# In[64]:


Hybrid_data = customerdata.drop(columns=['K_Means_cluster', 'Agglomerative_Cluster', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5','AcceptedCmp1', 'AcceptedCmp2','Complain', 'Response'])
Hybrid_data


# In[65]:


X = Hybrid_data.drop(columns=["Hybrid_Cluster"])
y = Hybrid_data['Hybrid_Cluster']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Fit and evaluate models
for model_name, model in classifiers.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = model.score(X_test, y_test)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"\033[1mModel Name: {model_name}...\033[0m")
    print(f"Accuracy of {model_name}: {accuracy:.2f}")
    print(f"Classification Report of {model_name}:\n{class_report}")
    print(f"Confusion Matrix of {model_name}:\n{conf_matrix}")
    print(f"AUC of {model_name}: {auc_score:.2f}")
    print(f"Cross-Validation Scores of {model_name}: {cv_scores}")
    print(f"Mean Cross-Validation Score of {model_name}: {cv_scores.mean():.2f}")
    print("--------------------------------------------------")

    print(f"\033[1mEvaluating {model_name}...\033[0m")
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)
    plot_precision_recall_curve(y_test, y_proba, model_name)


# # Summary
# Based on the evaluation metrics provided for the various models, here is a detailed summary of their performance and a recommendation for the best-fit model.
# 
# ### Logistic Regression
# - **Accuracy**: 0.87
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.78, 0.79, 0.79
#   - Class 1: 0.96, 0.97, 0.97
#   - Class 2: 0.77, 0.73, 0.75
# - **Confusion Matrix**:
#   ```
#   [[108   7  21]
#    [  7 210   0]
#    [ 24   1  69]]
#   ```
# - **AUC**: 0.96
# - **Cross-Validation Score**: 0.86
# 
# ### Support Vector Machine (SVM)
# - **Accuracy**: 0.79
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.66, 0.71, 0.69
#   - Class 1: 0.89, 0.92, 0.90
#   - Class 2: 0.75, 0.61, 0.67
# - **Confusion Matrix**:
#   ```
#   [[ 97  20  19]
#    [ 18 199   0]
#    [ 32   5  57]]
#   ```
# - **AUC**: 0.91
# - **Cross-Validation Score**: 0.78
# 
# ### Random Forest
# - **Accuracy**: 0.98
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.97, 0.98, 0.97
#   - Class 1: 0.99, 0.99, 0.99
#   - Class 2: 0.97, 0.96, 0.96
# - **Confusion Matrix**:
#   ```
#   [[133   1   2]
#    [  1 215   1]
#    [  3   1  90]]
#   ```
# - **AUC**: 1.00
# - **Cross-Validation Score**: 0.97
# 
# ### Gradient Boosting
# - **Accuracy**: 0.98
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.93, 0.99, 0.96
#   - Class 1: 1.00, 0.99, 0.99
#   - Class 2: 1.00, 0.91, 0.96
# - **Confusion Matrix**:
#   ```
#   [[135   1   0]
#    [  2 215   0]
#    [  8   0  86]]
#   ```
# - **AUC**: 1.00
# - **Cross-Validation Score**: 0.97
# 
# ### K-Nearest Neighbors (KNN)
# - **Accuracy**: 0.85
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.77, 0.76, 0.76
#   - Class 1: 0.92, 0.96, 0.94
#   - Class 2: 0.79, 0.73, 0.76
# - **Confusion Matrix**:
#   ```
#   [[103  15  18]
#    [  8 209   0]
#    [ 23   2  69]]
#   ```
# - **AUC**: 0.94
# - **Cross-Validation Score**: 0.85
# 
# ### Naive Bayes
# - **Accuracy**: 0.94
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.88, 0.94, 0.91
#   - Class 1: 1.00, 0.96, 0.98
#   - Class 2: 0.92, 0.89, 0.91
# - **Confusion Matrix**:
#   ```
#   [[128   1   7]
#    [  8 209   0]
#    [ 10   0  84]]
#   ```
# - **AUC**: 0.99
# - **Cross-Validation Score**: 0.95
# 
# ### Decision Tree
# - **Accuracy**: 0.96
# - **Precision, Recall, F1-Score**:
#   - Class 0: 0.93, 0.96, 0.94
#   - Class 1: 0.99, 0.99, 0.99
#   - Class 2: 0.95, 0.91, 0.93
# - **Confusion Matrix**:
#   ```
#   [[130   2   4]
#    [  2 214   1]
#    [  8   0  86]]
#   ```
# - **AUC**: 0.97
# - **Cross-Validation Score**: 0.96
# 
# ### Summary and Best Fit Model
# Based on the evaluation metrics, including accuracy, precision, recall, F1-score, AUC, and cross-validation scores, **Gradient Boosting** and **Random Forest** emerged as the top performers. Both models achieved a high accuracy of 0.98, perfect AUC scores of 1.00, and excellent cross-validation scores of 0.97.
# 
# Given the very high accuracy and consistent performance across all metrics, either of these models would be suitable for deployment. However, if we had to choose one, **Gradient Boosting** could be preferred due to its slightly better recall and F1-score for class 0 and 2, indicating better handling of minority classes.
# 
# ### Recommendation: Gradient Boosting
# - **Accuracy**: 0.98
# - **AUC**: 1.00
# - **Cross-Validation Score**: 0.97
# - **Key Strengths**:
#   - High precision, recall, and F1-scores across all classes.
#   - Perfect AUC, indicating excellent discriminatory power.
#   - Robust cross-validation performance, ensuring generalizability.
# 
# Thus, Gradient Boosting is recommended as the best-fit model for this classification task.

# # Hence we will deploy model using Gradient Boosting classifier as two clusters technique giving 0.99 accuracy .

# In[ ]:


#pip install scikit-learn pandas joblib


# In[75]:


Agglomerative_data.columns


# In[70]:


# Specify your feature columns and target column
X = Agglomerative_data.drop(columns=["Agglomerative_Cluster"])
y = Agglomerative_data['Agglomerative_Cluster']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[71]:


from sklearn.pipeline import make_pipeline
import joblib
# Create a pipeline with a standard scaler and a gradient boosting classifier
pipeline = make_pipeline(StandardScaler(), GradientBoostingClassifier())


# In[72]:


# Train the model
pipeline.fit(X_train, y_train)


# In[73]:


# Save the trained model to a file
joblib.dump(pipeline, 'gradient_boosting_model.pkl')

