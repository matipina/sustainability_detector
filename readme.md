# Course Similarity Finder

This Streamlit app helps you find semantically similar course descriptions based on a user-provided input. It uses a pre-trained Sentence-BERT model to compute semantic similarities between the input text and a dataset of course descriptions. The app allows you to filter the results based on a similarity threshold and offers options to hide duplicate descriptions and download the results as an Excel file.

## Features

- **Input Description**: Enter a text description to compare against the dataset of course descriptions.
- **Similarity Threshold**: Adjust the similarity threshold using a slider to control the level of matching.
- **Hide Duplicates**: Option to hide duplicate course descriptions in the results.
- **Download Results**: Download the filtered results as an Excel file.

## Installation

To run this app locally, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/course-similarity-app.git
    cd course-similarity-app
    ```

2. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your data**:
    - Place your processed data file (`processed_data.csv`) in the `data/` directory.
    - Place your pre-computed course embeddings file (`course_embeddings.npy`) in the `data/` directory.

4. **Run the app**:

    ```bash
    streamlit run main.py
    ```

## Usage

- **Enter a description** in the text area to compare it with course descriptions in the dataset.
- **Adjust the similarity threshold** using the slider to fine-tune the matching results. For a big list of keywords, I recommend around 0.3.
- **Download the results** as an Excel file by clicking the "Download" button.

## Acknowledgments

- The app uses the [Sentence-BERT](https://www.sbert.net) model for computing semantic similarities.