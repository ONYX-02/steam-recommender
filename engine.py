from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(game_name,svd_matrix, tfidf_matrix, game_titles, review_dict, name_to_id, top_n=10):
    
    if game_name not in game_titles:
        return []
    
    idx = game_titles.index(game_name)

    #Calculate similarity for just one game
    svd_sim = cosine_similarity(svd_matrix[idx:idx+1], svd_matrix).flatten()
    tfidf_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()

    # Blend the similarities
    hybrid_corr = (tfidf_sim * 0.6) + (svd_sim * 0.4)

    stop_words = {'the', 'of', 'and', 'in', 'a', 'to', 'for', 'edition', 'remaster'}
    target_words = set([w for w in game_name.lower().split() if w not in stop_words and not w.isnumeric()])

    scored_games = []

    for i, base_score in enumerate(hybrid_corr):
        sim_game = game_titles[i]
        
        if sim_game == game_name:
            continue

        sim_words = set([w for w in sim_game.lower().split() if w not in stop_words and not w.isnumeric()])
        if len(target_words.intersection(sim_words)) > 0: continue

        reviews = review_dict.get(sim_game, {'positive': 0, 'negative': 0})
        pos = reviews['positive']
        total = pos + reviews['negative']
        quality_ratio = (pos / total) if total > 0 else 0.70

        final_score = base_score * quality_ratio
        scored_games.append((sim_game, final_score, quality_ratio))

    scored_games.sort(key=lambda x: x[1], reverse=True)

    final_result = [] 
    for i in range(top_n):
        game, f_score, q_ratio = scored_games[i]
        app_id = name_to_id.get(str(game).lower().strip())

        final_result.append({
            'name': game,
            'app_id': app_id,
            'match_score': float(f_score),
            'rating': q_ratio
        })

    return final_result