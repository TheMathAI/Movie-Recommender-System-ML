# Movie-Recommender-System-Webapp-ML

# Description
The model ia a Collaborative Filtering recommendation system that was implemented using matrix factorization algorithm. The dataset we are using is MovieLens 1M dataset from Kaggle. https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset This dataset is very comprehensive and reliable. It contains 1,000,209 anonymous ratings of about 3,900 movies made by 6,040 MovieLens users with each user rated at least 20 movies.

The matrix factorization was done using singular value decomposition (svd), which can be implemented with the scipy library.

To train the model, k-fold cross-validation was used by masking different parts of the rating matrix. It was done with 200 different latent features (hyperparameter to decompose the rating matrix). 

To evaluate the accuracy of the model, the Frobenius norm was used. The best latent feature turns out to be 48.

Please look at the Training_Model file to see the code related to the data training.

# Webapp
To run the webapp make sure to download the following documents:
- app.py (file)
- templates (folder)
- clean_data (folder)

Now you only need to use the Flask library.

# Author
Félix Jean
