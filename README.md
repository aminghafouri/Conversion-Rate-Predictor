# Conversion Rate Prediction for Digital Marketing

## Project Objective
The objective is to build a machine learning model to predict the conversion rate of digital market listings (e.g., eBay). We are provided with two datasets, one for training the model and one for prediction. The features are comprised of a mix of continuous and categorical features. For each row in the test data, we need to provide a real value, specifying the conversion probability of an item.

## Imbalanced Data
eBay dataset is largely imbalanced. There are different approaches to address the sample imbalance problem: 
1. Balancing the training set through oversampling the minority class, under- sampling the majority class, or synthesizing new data for minority class. 
2. At the algorithm level through adjusting the class weight (i.e., misclassification cost), adjusting the decision threshold, or making the algorithm more sensitive to minority class.

## Preprocessing 
**Item Name & Description**: We assume names and descriptions are preprocessed, i.e., stemmed, lemmatized, removal of stop words and typos, etc. We can use Bag of Words (BOW) or term frequencyinverse document frequency (tf-idf) to represent item name and description. In a BOW approach, each element represents the frequency of a specific word. This frequency can be either boolean, indicating whether the word appears in the text or not, or numerical, indicating the number of occurrences in the text. The distance between the BOW vectors can be used to nd similar items. In tf-idf, a statistical measure is used to evaluate how important a word is to a document in a collection. In this project, we use tf-idf representation as it is more powerful and can achieve better results.

**Features**: Our prediction model is based upon the features item name, item description, number of clicks per sale, seller ID, item price, listing attributes We could also use item's fair market price. Note that to estimate item's fair market price, one can use similar items sold as it reveal important demand measurements. This can be highly beneficial since as stated in [1], the ratio of initial total price to the median total price of recently sold similar items is highly correlated with conversion rate. Seller ID is also considered as it is a strong indicator of conversion rate.

## Learning 
We use XGBoost, which is an optimized distributed gradient boosting method designed to be highly efficient, exible and portable. Gradient boosting methods are know to work well for non-linear heterogeneous classi cation/regression models with large number of factors. The result is stable even when some important factors are missing at runtime. Note that other approaches can also be used. For example, weighted random forest is also shown to perform well, especially if the out-of-bag estimate of the accuracy from random forest is used to select weights [2].

## Evaluation
We use 100 decision trees to train our predictive model. We select a small value for learning rate as small values of learning rate will help prevent the algorithm to over- t the training data. Further, validation sets are sampled without overlapping with the training data. 

Since the sample is imbalanced, the overall accuracy is not an appropriate measure of performance. This is because a trivial classi er that predicts every case as the majority class can still achieve very high accuracy. Therefore, we need use other performance metrics such as precision and recall, false positive rate and false negative rate, F-measure and weighted accuracy. In this report, we use the area under the ROC curve between the predicted probability and the observed target. 

Given the evaluation scenario described above, and by splitting the dataset with a 3:1 ratio, our method obtains an AUROC of 0.902. Moreover, for each row in the validation data, our code provides the conversion probability of the corresponding item.
