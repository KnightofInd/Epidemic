import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved model and data
model_file = './medicine.pkl'
data_file = './medicines_data.pkl'

# Load the TF-IDF model
tfidf = joblib.load(model_file)

# Load the dataframe and the vectorized matrix
df, tfidf_matrix = joblib.load(data_file)

def recommend_medicine(salt1, salt2=''):
    # Combine user input into one string
    query = salt1 + ' ' + salt2
    # Transform the input salt combination to the same TF-IDF vector space
    query_vec = tfidf.transform([query])
    
    # Compute cosine similarity between the input vector and all the medicine vectors
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    
    # Get the index of the most similar medicine
    idx = similarity.argsort()[0][-1]
    
    # Return the most similar medicine name
    return df['name'].iloc[idx]

# Example usage:
recommended_medicine = recommend_medicine('Amoxycillin (500mg)', 'Clavulanic Acid (125mg)')
print(f"Recommended Medicine: {recommended_medicine}")