import streamlit as st
import pickle
from engine import get_recommendations

st.set_page_config(page_title="Steam Game Recommender", layout="centered")

# Load .pkl files into memory
@st.cache_data
def load_data():
    with open('model/game_titles.pkl', 'rb') as f: game_titles = pickle.load(f)
    with open('model/review_dict.pkl', 'rb') as f: review_dict = pickle.load(f)
    with open('model/svd_matrix.pkl', 'rb') as f: svd_matrix = pickle.load(f)
    with open('model/tfidf_matrix.pkl', 'rb') as f: tfidf_matrix = pickle.load(f)
    with open('model/name_to_id.pkl', 'rb') as f: name_to_id = pickle.load(f)
    return game_titles, review_dict, svd_matrix, tfidf_matrix, name_to_id

game_titles, review_dict, svd_matrix, tfidf_matrix, name_to_id = load_data()

st.title("Steam Game Recommender")

selected_game = st.selectbox(
    "Search for a game:", 
    game_titles, 
    index=None, 
    placeholder="Search for a game", 
    label_visibility="collapsed")

if st.button("Get Recommendations"):
    with st.spinner("Calculating matrices..."):

        results = get_recommendations(
            game_name=selected_game,
            svd_matrix=svd_matrix,
            tfidf_matrix=tfidf_matrix,
            game_titles=game_titles,
            review_dict=review_dict,
            name_to_id=name_to_id
        )

        st.subheader(f"Recommendations for {selected_game}:")

        for game in results:
            col1, col2 = st.columns([1, 2])
            with col1:
                if game['app_id']:
                    img_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{game['app_id']}/header.jpg"
                    st.image(img_url, width="stretch")
                else:
                    st.write("*(No Image Availble)*")

            with col2:
                st.markdown(f"### {game['name']}")
                st.write(f"**Steam Rating:** {game['rating'] * 100:.1f}% Positive")
                st.write(f"**ML Engine Match Score:** {game['match_score'] * 100:.1f}%")

            st.divider()