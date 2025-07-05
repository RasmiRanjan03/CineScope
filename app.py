from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')

# Load movies data
movies_df = pd.read_pickle(r'E:\Programming\MovieRecommend\project2\movies_list.pkl')

# Clean and normalize data
movies_df['title_clean'] = movies_df['title'].str.lower().str.strip()
movies_df['tags_clean'] = movies_df['tags'].str.lower().str.strip()

# Create similarity matrix
similarity_matrix = pd.read_pickle(r'E:\Programming\MovieRecommend\project2\similarlity.pkl')

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/search', methods=['POST'])
def search_movies():
    try:
        data = request.json
        search_term = data.get('movie_name', '').lower().strip()
        
        if not search_term:
            # Return top rated movies if no search term
            top_movies = movies_df.sort_values(['vote_average', 'vote_count'], ascending=False).head(20)
            recommended_movies = []
            for _, row in top_movies.iterrows():
                recommended_movies.append({
                    'title': str(row.get('title', '')),
                    'description': str(row.get('tags', '')),
                    'popularity': float(row.get('popularity', 0)),
                    'vote_average': float(row.get('vote_average', 0)),
                    'vote_count': int(row.get('vote_count', 0)),
                    'score': float(row.get('vote_average', 0)) / 10.0  # Normalize to 0-1 scale
                })
            return jsonify({
                'status': 'success',
                'recommendations': recommended_movies
            })
        
        # Search in titles and tags
        title_matches = movies_df[movies_df['title_clean'].str.contains(search_term, na=False)]
        tag_matches = movies_df[movies_df['tags_clean'].str.contains(search_term, na=False)]
        
        # Combine and remove duplicates
        all_matches = pd.concat([title_matches, tag_matches]).drop_duplicates()
        
        # Sort by relevance (title matches first, then by popularity and rating)
        all_matches['relevance_score'] = 0
        title_exact_matches = all_matches[all_matches['title_clean'].str.contains(search_term, na=False)]
        all_matches.loc[title_exact_matches.index, 'relevance_score'] += 10
        
        # Add popularity and rating to relevance score
        all_matches['relevance_score'] += all_matches['popularity'] * 0.001 + all_matches['vote_average'] * 0.1
        
        # Sort by relevance score
        all_matches = all_matches.sort_values('relevance_score', ascending=False)
        
        # Format recommendations
        recommended_movies = []
        for _, row in all_matches.head(20).iterrows():
            recommended_movies.append({
                'title': str(row.get('title', '')),
                'description': str(row.get('tags', '')),
                'popularity': float(row.get('popularity', 0)),
                'vote_average': float(row.get('vote_average', 0)),
                'vote_count': int(row.get('vote_count', 0)),
                'score': min(float(row.get('relevance_score', 0)) / 10.0, 1.0)  # Normalize to 0-1 scale
            })
        
        return jsonify({
            'status': 'success',
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        
        # Extract user preferences
        user_age = int(data.get('userAge', 25))
        favorite_actor = data.get('favoriteActor', '').lower()
        favorite_actress = data.get('favoriteActress', '').lower()
        preferred_genre = data.get('preferredGenre', '').lower()
        
        # Calculate similarity scores
        movies_df['similarity_score'] = movies_df.apply(lambda x: 
            (favorite_actor in str(x.get('title', '')).lower() or 
             favorite_actor in str(x.get('tags', '')).lower()) * 0.4 +
            (favorite_actress in str(x.get('title', '')).lower() or 
             favorite_actress in str(x.get('tags', '')).lower()) * 0.3 +
            (preferred_genre in str(x.get('tags', '')).lower()) * 0.3 +
            float(x.get('popularity', 0)) * 0.001 +
            float(x.get('vote_average', 0)) * 0.01 +
            float(x.get('vote_count', 0)) * 0.00001,
            axis=1
        )
        
        # Get top recommendations
        recommendations = movies_df.sort_values('similarity_score', ascending=False)
        recommendations = recommendations[recommendations['similarity_score'] > 0]
        
        # If no recommendations found, try a more lenient search
        if len(recommendations) == 0:
            movies_df['lenient_score'] = movies_df.apply(lambda x: 
                sum(1 for word in str(x.get('title', '')).lower().split() 
                    if favorite_actor in word or favorite_actress in word) * 0.1 +
                sum(1 for word in str(x.get('tags', '')).lower().split() 
                    if favorite_actor in word or favorite_actress in word or preferred_genre in word) * 0.1,
                axis=1
            )
            recommendations = movies_df.sort_values('lenient_score', ascending=False)
            recommendations = recommendations[recommendations['lenient_score'] > 0]
            
            # If still no recommendations, return top rated movies
            if len(recommendations) == 0:
                recommendations = movies_df.sort_values(['vote_average', 'vote_count'], 
                                                      ascending=False).head(20)
        
        # Format recommendations
        recommended_movies = []
        for _, row in recommendations.head(20).iterrows():
            recommended_movies.append({
                'title': str(row.get('title', '')),
                'description': str(row.get('tags', '')),
                'popularity': float(row.get('popularity', 0)),
                'vote_average': float(row.get('vote_average', 0)),
                'vote_count': int(row.get('vote_count', 0)),
                'score': min(float(row.get('similarity_score', 0)) / 10.0, 1.0)
            })
        
        return jsonify({
            'status': 'success',
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
