import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Title
st.title("Big Firms & Startups Matchmaking")

# Sidebar for File Uploads
st.sidebar.header("Upload Datasets")
big_firms_file = st.sidebar.file_uploader("Upload Big Firms Dataset (CSV)", type=["csv"])
startups_file = st.sidebar.file_uploader("Upload Startups Dataset (CSV)", type=["csv"])

# Slider for Top Matches
top_k = st.sidebar.slider("Select Top Matches", min_value=1, max_value=20, value=5)

# Process Uploaded Data
if big_firms_file and startups_file:
    try:
        big_firms = pd.read_csv(big_firms_file)
        startups = pd.read_csv(startups_file)

        # Check if required columns exist
        required_columns = ['Company Name', 'Technology', 'Research and Development Area']
        for col in required_columns:
            if col not in big_firms.columns:
                st.error(f"Big Firms dataset is missing the required column: {col}")
                st.stop()
            if col not in startups.columns:
                st.error(f"Startups dataset is missing the required column: {col}")
                st.stop()

        # Ensure top_k does not exceed the number of startups
        if top_k > len(startups):
            st.error(f"The selected number of top matches ({top_k}) exceeds the number of available startups ({len(startups)}). Please select a smaller number.")
            st.stop()

        # Display uploaded datasets
        st.subheader("Big Firms Dataset")
        st.dataframe(big_firms)

        st.subheader("Startups Dataset")
        st.dataframe(startups)

        # Prepare Text for Semantic Search
        big_firm_texts = (
            big_firms['Technology'] + " " + big_firms['Research and Development Area']
        ).fillna("N/A").tolist()
        startup_texts = (
            startups['Technology'] + " " + startups['Research and Development Area']
        ).fillna("N/A").tolist()

        # Encode using SentenceTransformer
        big_firm_embeddings = model.encode(big_firm_texts, convert_to_tensor=True)
        startup_embeddings = model.encode(startup_texts, convert_to_tensor=True)

        # Calculate similarity scores
        similarity_scores = util.pytorch_cos_sim(big_firm_embeddings, startup_embeddings)

        # Find Top Matches
        results = []
        for i, firm in enumerate(big_firms['Company Name']):
            scores = similarity_scores[i]
            top_matches = torch.topk(scores, top_k)
            for idx in top_matches.indices:
                idx = int(idx)  # Ensure integer indexing
                results.append({
                    "Big Firm": firm,
                    "Startup": startups.iloc[idx]['Company Name'],
                    "Similarity Score": scores[idx].item()
                })

        # Display Results
        results_df = pd.DataFrame(results)

        # Sort results by "Big Firm" and "Similarity Score"
        sorted_results = results_df.sort_values(by=["Big Firm", "Similarity Score"], ascending=[True, False])

        # Display sorted results
        st.subheader("Matchmaking Results (Sorted)")
        st.dataframe(sorted_results)

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
else:
    st.info("Please upload both Big Firms and Startups datasets to proceed.")
