# import library
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

df_movies = pd.read_csv("./clean_data/df_movies.csv")
df_ratings = pd.read_csv("./clean_data/df_ratings.csv")

# making the rating dataframe with column for column for movies and row for users (kind of a matrix)
df_ratings_pivot = df_ratings.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)

# turning the datafram into a numpy array for calculations
df_ratings_array = df_ratings_pivot.values
# calculating the mean for each users
mean_users_ratings = np.mean(df_ratings_array, axis = 1)
# demeaning the the array, so that each observations are mean zero
df_demean = df_ratings_array - mean_users_ratings.reshape(-1, 1)

# singular value decomposition of the matrix, to turn the matrix into a singular vector with singular values
U, s, Vh = svds(df_demean, k = 48)

# diagonalize matrix
s = np.diag(s)

# get the user predicting ratings for movies and then add the mean back to get right ratings
predicted_ratings = np.dot(np.dot(U, s), Vh) + mean_users_ratings.reshape(-1, 1)
# turn the predicted ratings into data frame
df_predictions = pd.DataFrame(predicted_ratings, columns = df_ratings_pivot.columns)

def recommend_movies(df_predictions, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 
    sorted_user_predictions = df_predictions.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieID', right_on = 'movieID').sort_values(['rating'], ascending=False))

    print(f'User {userID} has rated {user_full.shape[0]} movies.')
    print(f'Recommending the highest {num_recommendations} predicted ratings movies not already rated.')
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (df_movies[~df_movies['movieID'].isin(user_full['movieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieID',
               right_on = 'movieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

user_ratings, user_predictions = recommend_movies(df_predictions, 837, df_movies, df_ratings, 10)

print(user_predictions)