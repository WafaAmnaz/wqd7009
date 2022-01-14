#Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st 

st.write("USING SINGULAR VALUE DECOMPOSITION (SVD) IN A RECOMMENDER SYSTEMS")
st.write("-------------------------------------------------------------")
st.write(" ")
st.write("Member of Group 7:")
st.write("1. Baharul Hisyam bin Baharudin (S2039609)")
st.write("2. Nor Azyra Binti Omar (17120332)")
st.write("3. Nur Farzanah Roslan (17089384)")
st.write("4. Wardatul Fadhilah binti Amir Nazri (S2039977)")


#Ignore the error
np.seterr(divide='ignore', invalid='ignore')


# streamlit run SVD.py


st.write(" ")
st.write("Reading dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)")
with st.echo():
    data = pd.io.parsers.read_csv('data/ratings.dat', 
        names=['user_id', 'movie_id', 'rating', 'time'],
        engine='python', delimiter='::')
    movie_data = pd.io.parsers.read_csv('data/movies.dat',
        names=['movie_id', 'title', 'genre'],
        engine='python', delimiter='::')

st.write(" ")
st.write("Display partial data from ratings table")
st.write(data[:5])
st.write(" ")
st.write("Display partial data from movies table")
st.write(movie_data[:5])

st.write(" ")
st.write("Creating the rating matrix (rows as movies, columns as users)")
with st.echo():
    ratings_mat = np.ndarray(
        shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
        dtype=np.uint8)
    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

st.write(" ")
st.write("st.write(" ")Normalizing the matrix(subtract mean off)")
with st.echo():
    normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

st.write(" ")
st.write("Computing the Singular Value Decomposition (SVD)")
with st.echo():
    A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
    U, S, V = np.linalg.svd(A)

st.write(" ")
st.write("Function to calculate the cosine similarity (sorting by most similar")
with st.echo():
    def top_cosine_similarity(data, movie_id, top_n=10, bottom_n=3):
        """ 
        Function to calculate the cosine similarity (sorting by most similar
        and returning the top N)
                
        """
        index = movie_id - 1 # Movie id starts from 1 in the dataset
        movie_row = data[index, :]
        magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))

        similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)

        sort_indexes = np.argsort(-similarity)
        return sort_indexes[:top_n], sort_indexes[-bottom_n:]

st.write(" ")
st.write("Function to print top N similar movies and M least similar movies")
with st.echo():
    def print_similar_movies(movie_data, movie_id, top_indexes, bottom_indexes):
        """
        Function to print top N similar movies and M least similar movies
        """
        st.write("\n\n********************************************")
        st.write("------------------------------------------------")
        st.write('Recommendations for {0}: \n'.format(
        movie_data[movie_data.movie_id == movie_id].title.values[0]))
        st.write("------------------------------------------------")
        for id in top_indexes + 1:
            st.write(movie_data[movie_data.movie_id == id].title.values[0] + ", Genre: " + movie_data[movie_data.movie_id == id].genre.values[0])  

        st.write("\n\n------------------------------------------------")
        st.write('Least similar to {0}: \n'.format(
        movie_data[movie_data.movie_id == movie_id].title.values[0]))
        st.write("------------------------------------------------")
        for id in bottom_indexes:
            st.write(movie_data[movie_data.movie_id == id].title.values[0] + ", Genre: " + movie_data[movie_data.movie_id == id].genre.values[0])           


st.write("Get recommendations for three movies")
with st.echo():
    #k-principal components to represent movies, movie_id to find recommendations,
    #top_n : print n results
    k = 50
    top_n = 10
    bottom_n = 3
    sliced = V.T[:, :k] # representative data

    movie_id_list = [3793,  # X-Men (2000)
                     2808,  # Universal Soldier (1992)
                     10]    # GoldeEye (1995)

    for movie_id in movie_id_list:
        similar_indexes, least_similar_indexes = top_cosine_similarity(sliced, movie_id, top_n, bottom_n)
        
        print_similar_movies(movie_data, movie_id, similar_indexes, least_similar_indexes)


