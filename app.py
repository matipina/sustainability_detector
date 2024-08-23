import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import io


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

# Option to upload a CSV file if not found in the default path
uploaded_file = st.file_uploader("Upload your processed data CSV file. If no file is uploaded, the default path will be used.", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('data/processed_data.csv')

# Inputs
input_text = st.text_area("Enter a text to compare. This can be a description, a list of keywords, or anything else:", "")
threshold = st.slider("Similarity Threshold (adjust depending on input)", 0.0, 1.0, 0.3)
output_path = st.text_input("Enter output file path (optional):", "")

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

course_embeddings = torch.tensor(np.load('data/course_embeddings.npy')).to(device)
    
# Button to run the similarity search
if st.button("Find Similar Courses"):
    with st.spinner("Computing similarities..."):
        similar_courses_df = find_semantically_similar_courses(input_text, course_embeddings, df, model, threshold, device=device)
        st.session_state['similar_courses_df'] = similar_courses_df

# Retrieve the results from session state
if 'similar_courses_df' in st.session_state:
    similar_courses_df = st.session_state['similar_courses_df']
    st.write(f"Found {len(similar_courses_df)} results.")
    
    # Checkbox to hide duplicates
    hide_duplicates = st.checkbox("Hide duplicate descriptions")
    if hide_duplicates:
        similar_courses_df = similar_courses_df.drop_duplicates(subset=['Course Description'])

    st.dataframe(similar_courses_df[['Course Title', 'Term ', 'CRN', 'Course Description', 'Similarity']])

    # Save results if output path is provided
    if output_path:
        st.write(f"Saving results to {output_path}...")
        similar_courses_df.to_excel(output_path, sheet_name='Similar Courses', index=False)
        st.write("Results saved successfully!")

    # Download button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        similar_courses_df.to_excel(writer, sheet_name='Similar Courses', index=False)
    
    st.download_button(
        label="Download results as Excel",
        data=buffer.getvalue(),
        file_name='similar_courses.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )