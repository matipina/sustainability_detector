import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch


def find_semantically_similar_courses(input_text, course_embeddings, df, model, threshold=0.7, device='cpu'):
    # Compute embeddings for the input text
    input_embedding = model.encode(input_text, convert_to_tensor=True, device=device)
    
    # Ensure both input and course embeddings are on the same device
    course_embeddings = course_embeddings.to(device)

    # Compute cosine similarities between the input text and each course description
    similarities = util.pytorch_cos_sim(input_embedding, course_embeddings)[0]
    
    # Filter results based on the similarity threshold
    df['Similarity'] = similarities.cpu().numpy()
    similar_courses_df = df[df['Similarity'] >= threshold].sort_values(by='Similarity', ascending=False)
    
    return similar_courses_df

# Streamlit app
st.title("Course Description Similarity Finder")

# Inputs
input_text = st.text_area("Enter a description to compare:", "This is sample text. Replace with whatever is needed.")
threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5)
output_path = st.text_input("Enter output file path (optional, to save results):", "")

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

course_embeddings = torch.tensor(np.load('data/course_embeddings.npy')).to(device)
df = pd.read_csv('data/processed_data.csv')

# Button to run the similarity search
if st.button("Find Similar Courses"):
    with st.spinner("Computing similarities..."):
        similar_courses_df = find_semantically_similar_courses(input_text, course_embeddings, df, model, threshold, device=device)

    st.write(f"Found {len(similar_courses_df)} results.")
    st.dataframe(similar_courses_df[['Course Title', 'Term ', 'CRN', 'Course Description', 'Similarity']])

    if output_path:
        st.write(f"Saving results to {output_path}...")
        similar_courses_df.to_excel(output_path, sheet_name='Similar Courses', index=False)
        st.write("Results saved successfully!")

