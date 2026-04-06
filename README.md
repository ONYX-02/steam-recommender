# Steam Game Recommender

A machine learning-powered web application that recommends Steam games based on your favorites. 

Instead of basic genre matching, this engine uses a **Hybrid Recommendation System** combining Natural Language Processing (TF-IDF on game tags/descriptions) and Collaborative Filtering (Truncated SVD) to find deep, nuanced similarities between over 15,000 games. It instantly calculates matches on the fly and fetches live header images directly from the Steam CDN.

## Features
* **Hybrid ML Engine:** Blends TF-IDF and SVD algorithms for high-quality recommendations.
* **Smart Filtering:** Automatically detects and filters out direct sequels/prequels to force the algorithm to find *new* franchises.
* **Quality Control:** Penalizes games with "Mixed" or "Negative" Steam review ratios, bubbling up "Overwhelmingly Positive" hidden gems.
* **On-The-Fly Computation:** Matrix math is calculated dynamically in milliseconds, bypassing the need for massive $O(N^2)$ gigabyte storage files.
* **Clean UI:** Built with Streamlit for a fast, responsive, and visual user experience.

## Project Structure

The repository is built with a clean Separation of Concerns:

'''text
STEAM-RECOMMENDER/
│
├── app.py                 # Streamlit frontend UI & caching
├── engine.py              # Backend machine learning logic & math
├── requirements.txt       # Python dependencies (Streamlit, scikit-learn, etc.)
│
├── model/                 # Compiled, lightweight base matrices
│   ├── game_titles.pkl
│   ├── name_to_id.pkl
│   ├── review_dict.pkl
│   ├── svd_matrix.pkl
│   └── tfidf_matrix.pkl
│
└── notebook/              # ETL pipelines and model training
    ├── process_dataset.ipynb
    ├── scraper.ipynb
    └── train_model.ipynb

## Tech Stack
* **Language** Python
* **Data Processing** Pandas, NumPy
* **Machine Learning** Scikit-Learn (TruncatedSVD, TfidfVectorizer, Cosine Similarity)
* **Data Ingestion** Requests, JSON, Steam API / SteamSpy API

## How to Run Locally
Note: The raw CSV datasets are not included in this repository to save space. The web app runs entirely off the pre-processed `.pkl` matrices in the `model/` folder. If you wish to re-train the model from scratch, you will need to run the web scraper notebook to generate a fresh dataset.
1. Clone the repository.
2. (Optional, but recommended) Set up the virtual environment
3. Install dependencies:
   - pip install streamlit pandas numpy scikit-learn
4. From the "steam-recommender" folder, run the following command:
   - streamlit run app.py

## Future Improvements
* Integrate the official Steam API to fetch more titles and bypass the dataset limitation.