# Steam Game Hybrid Recommendation Engine

An end-to-end data pipeline and hybrid recommendation system that suggests Steam games based on both **player behavior** (Collaborative Filtering) and
**game tags** (Content-Based NLP).

## Project Overview
Most recommendation engines only use either content only or behavior only. This project solves both by blending mathematical matrices from two distinct
data sources:
1. **User Interaction Data** Processed from a massive dataset of Steam player reviews and playtimes using memory-efficient chunking.
   (Dataset Link: https://www.kaggle.com/datasets/kieranpoc/steam-reviews)
2. **Community Metadata** Scraped dynamically from the SteamSpy API to capture tags, review scores and genres.

## The Architecture
### 1. The Collaborative Brain (SVD)
* Built a sparse User-Item matrix mapping players to hours played.
* Applied a **Logorithmic Transformation ('np.log1p')** to normalize playtime, to prevent games with many hours played from skewing geometry.
* Reduced dimensionality using **TruncatedSVD (Singular Value Decomposition)** to uncover hidden player archetypes

### 2. The Content Brain (NLP)
* Utilized **TF-IDF (Term Frequency-Inverse Document Frequency)** on the "bag of words" from SteamSpy community tags.
* Transforms game mechanics (e.g., "Post-Apocalyptic", "RPG") into spatial vectors to measure cosine similarity between titles.

### 3. The Custom Data Pipeline & Web Scraper
* Built a custom Python scraper with error handling and request throttling to extract metadata from SteamSpy.
* **NSFW/Troll Filter** Engineered a "Top-4 Tag" filter to dynamically detect and ban explicitly mature games, while bypassing community "troll tags" on innocent games (like LEGO titles).

### 4. Algorithmic Guardrails
* **The Sequel Filter** NLP-based title parsing that prevents the engine from recommending games from the same franchise (e.g., searching for *Fallout 4* won't just return *Fallout: New Vegas*).
* **The Quality Penalty** Dynamically scales the final Hybrid Score against the game's official Steam Approval Rating, pushing poorly reviewed games to the bottom of the recommendations.

## Tech Stack
* **Language** Python
* **Data Processing** Pandas, NumPy
* **Machine Learning** Scikit-Learn (TruncatedSVD, TfidfVectorizer, Cosine Similarity)
* **Data Ingestion** Requests, JSON, Steam API / SteamSpy API

## How to Run Locally
Note: The raw Steam user dataset is too large to host on GitHub, but the final scraped metdata (tags, review scores) is included as JSON files so you can run the
train_model.ipynb file immediately
1. Clone the repository.
2. Download the Steam Reviews dataset (link above) to a folder named "data" in the base directory.
2. Ensure you have your `.env` file configured with your local paths/keys.
3. Run the data chunking script to compress the raw Steam dataset.
4. Run the Master Scraper to build the local metadata JSONs.
5. Execute the Jupyter Notebook to build the matrices and run the `get_recommendations()` function.

## Future Improvements
* Build a frontend for real-time user interaction.
* Integrate the official Steam API to fetch modern (2024+) titles and bypass the 2022 dataset limitation.