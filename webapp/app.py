from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        one = request.form.get("1")
        one107 = request.form.get("107")
        one254 = request.form.get("254")
        one288 = request.form.get("288")
        one310 = request.form.get("310")
        one347 = request.form.get("347")
        one467 = request.form.get("467")
        one514 = request.form.get("514")
        one576 = request.form.get("576")
        one580 = request.form.get("580")
        one594 = request.form.get("594")
        one803 = request.form.get("803")
        one1026 = request.form.get("1026")
        one1107 = request.form.get("1107")
        one2786= request.form.get("2786")
        one2558 = request.form.get("2558")
        one1121 = request.form.get("1121")
        one1149 = request.form.get("1149")
        list_rate = [one, one107, one254, one288, one310, one347, one467, one514, one576, one580,one594,one803,one1026,one1107,one2786,one2558,one1121,one1149]
        m_id=  [1, 107, 254, 288, 310, 347, 467, 514, 576, 580,594,803,1026,1107,2786,2558,1121,1149]
        n = 18
        # Put inputs to dataframe
        X = pd.Series(list_rate, index= m_id)
        
        

        # load dataframe
        df_movies = pd.read_csv("./clean_data/df_movies.csv")
        df_ratings = pd.read_csv("./clean_data/df_ratings.csv")
        list_df = list(map(str, set(list(df_ratings['movieID']))))
        # pivot df
        df_ratings_pivot = df_ratings.pivot(index = 'userID', columns ='movieID', values = 'rating').fillna(0)
        df_ratings_pivot = pd.concat([df_ratings_pivot, X.to_frame().T], ignore_index=True).fillna(0)
    
        df_ratings_pivot.set_axis(list_df, axis=1,inplace=True)
        Y = {'userID': [(df_ratings_pivot.shape)[0]]*n,
        'movieID': m_id, 'rating': list_rate}

        df_ratings = pd.concat([df_ratings,pd.DataFrame(Y)], ignore_index=True)

        df_ratings_pivot = df_ratings_pivot.astype('float')
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
        list_df = list(map(int, list_df))
        df_predictions.set_axis(list_df, axis=1,inplace=True)
        df_predictions.columns.name = 'movieID' 
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
        user_ratings, user_predictions = recommend_movies(df_predictions, (df_ratings_pivot.shape)[0], df_movies, df_ratings, 10)
        
        predictions = user_predictions.iloc[:,0:2]
        # Get prediction just trying to print "hi" for the moment
        return render_template("website.html", output =[predictions.to_html(classes='data')], titles=predictions.columns.values)
        
    else:
        return render_template("website.html", ouput = pd.DataFrame(["0"], columns=['No Recommendation']))
        
    

# Running the app
if __name__ == '__main__':
    app.run(debug = True)