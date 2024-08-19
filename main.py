import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import argparse


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

def main(input_text, threshold, output_path):
    # Load the pre-trained Sentence-BERT model and course embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Load the course embeddings and move them to the same device
    course_embeddings = torch.tensor(np.load('data/course_embeddings.npy')).to(device)

    print('Reading data...')
    df = pd.read_csv('data/processed_data.csv')

    print(f'Running on device: {device}.')
    print(f'Computing text similarity of input: \n\n{input_text}\n')
    similar_courses_df = find_semantically_similar_courses(input_text, course_embeddings, df, model, threshold, device=device)

    # Display the similar courses
    print(f'Found {len(similar_courses_df)} results.')
    print(similar_courses_df[['Term ', 'CRN', 'Course Title', 'Similarity']])
    
    if output_path:
        print(f'Saving results to {output_path}...')
        similar_courses_df.to_excel(output_path, sheet_name='Similar Courses', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find semantically similar courses.')
    
    # Add input text argument (optional)
    parser.add_argument('--input_text', type=str, help='The text to compare with the course descriptions.')
    
    # Add threshold argument (optional)
    parser.add_argument('--threshold', type=float, default=0.5, help='The similarity threshold for filtering results.')
    
    # Add output file argument (optional)
    parser.add_argument('--output_path', type=str, default=None, help='Path to the output file.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set the input text and threshold
    default_text = "This is sample text. Replace with whatever is needed."
    input_text = args.input_text if args.input_text else default_text
    threshold = args.threshold
    output_path = args.output_path
    
    # Run the main function
    main(input_text, threshold, output_path)
