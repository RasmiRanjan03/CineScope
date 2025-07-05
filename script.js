// Movie database with sample data
const movieDatabase = {
    action: [
        {
            title: "The Dark Knight",
            year: 2008,
            genre: "Action, Crime, Drama",
            rating: 9.0,
            plot: "When the menace known as the Joker wreaks havoc on Gotham City, Batman must accept one of the greatest psychological tests of his ability to fight injustice.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        },
        {
            title: "Mad Max: Fury Road",
            year: 2015,
            genre: "Action, Adventure, Sci-Fi",
            rating: 8.1,
            plot: "In a post-apocalyptic wasteland, Max teams up with a mysterious woman to try to survive.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        },
        {
            title: "John Wick",
            year: 2014,
            genre: "Action, Crime, Thriller",
            rating: 7.4,
            plot: "An ex-hitman comes out of retirement to track down the gangsters that took everything from him.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        }
    ],
    romantic: [
        {
            title: "The Notebook",
            year: 2004,
            genre: "Drama, Romance",
            rating: 7.8,
            plot: "A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom.",
            poster: "https://images.unsplash.com/photo-1721322800607-8c38375eef04?w=300&h=200&fit=crop"
        },
        {
            title: "Titanic",
            year: 1997,
            genre: "Drama, Romance",
            rating: 7.9,
            plot: "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious ship.",
            poster: "https://images.unsplash.com/photo-1721322800607-8c38375eef04?w=300&h=200&fit=crop"
        },
        {
            title: "La La Land",
            year: 2016,
            genre: "Comedy, Drama, Music, Romance",
            rating: 8.0,
            plot: "A jazz musician and an aspiring actress meet and fall in love in Los Angeles.",
            poster: "https://images.unsplash.com/photo-1721322800607-8c38375eef04?w=300&h=200&fit=crop"
        }
    ],
    drama: [
        {
            title: "The Shawshank Redemption",
            year: 1994,
            genre: "Drama",
            rating: 9.3,
            plot: "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
            poster: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=300&h=200&fit=crop"
        },
        {
            title: "Forrest Gump",
            year: 1994,
            genre: "Drama, Romance",
            rating: 8.8,
            plot: "The presidencies of Kennedy and Johnson through the eyes of an Alabama man with an IQ of 75.",
            poster: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=300&h=200&fit=crop"
        },
        {
            title: "The Godfather",
            year: 1972,
            genre: "Crime, Drama",
            rating: 9.2,
            plot: "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
            poster: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=300&h=200&fit=crop"
        }
    ],
    comedy: [
        {
            title: "The Grand Budapest Hotel",
            year: 2014,
            genre: "Adventure, Comedy, Crime",
            rating: 8.1,
            plot: "A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy.",
            poster: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=300&h=200&fit=crop"
        },
        {
            title: "Superbad",
            year: 2007,
            genre: "Comedy",
            rating: 7.6,
            plot: "Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry.",
            poster: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=300&h=200&fit=crop"
        }
    ],
    thriller: [
        {
            title: "Inception",
            year: 2010,
            genre: "Action, Sci-Fi, Thriller",
            rating: 8.8,
            plot: "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        },
        {
            title: "Se7en",
            year: 1995,
            genre: "Crime, Drama, Mystery, Thriller",
            rating: 8.6,
            plot: "Two detectives hunt a serial killer who uses the seven deadly sins as his motives.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        }
    ],
    horror: [
        {
            title: "The Exorcist",
            year: 1973,
            genre: "Horror",
            rating: 8.1,
            plot: "When a teenage girl is possessed by a mysterious entity, her mother seeks the help of two priests to save her daughter.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        },
        {
            title: "Hereditary",
            year: 2018,
            genre: "Drama, Horror, Mystery, Thriller",
            rating: 7.3,
            plot: "A grieving family is haunted by tragedy and disturbing secrets.",
            poster: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=300&h=200&fit=crop"
        }
    ],
    "sci-fi": [
        {
            title: "Interstellar",
            year: 2014,
            genre: "Adventure, Drama, Sci-Fi",
            rating: 8.6,
            plot: "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
            poster: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=300&h=200&fit=crop"
        },
        {
            title: "Blade Runner 2049",
            year: 2017,
            genre: "Action, Drama, Mystery, Sci-Fi, Thriller",
            rating: 8.0,
            plot: "A young blade runner's discovery of a long-buried secret leads him to track down former blade runner Rick Deckard.",
            poster: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=300&h=200&fit=crop"
        }
    ]
};

// Movie recommendation system
let userPreferences = {};
let currentMovies = [];

async function getRecommendations(movieName) {
    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ movie_name: movieName })
        });
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        return data.recommendations;
    } catch (error) {
        console.error('Error getting recommendations:', error);
        return [];
    }
}

// Initialize with some default movies on page load
window.addEventListener('load', function() {
    console.log('Movie Recommendation System Loaded');
    // Don't show any movies initially - wait for user form submission
});

// Form submission handler
document.getElementById('preferencesForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get user preferences
    userPreferences = {
        name: document.getElementById('userName').value,
        age: parseInt(document.getElementById('userAge').value),
        favoriteActor: document.getElementById('favoriteActor').value,
        favoriteActress: document.getElementById('favoriteActress').value,
        preferredGenre: document.getElementById('preferredGenre').value
    };
    
    // Hide form and show recommendations
    document.getElementById('userForm').style.display = 'none';
    document.getElementById('recommendationsSection').style.display = 'block';
    
    // Display user name
    document.getElementById('displayName').textContent = userPreferences.name;
    
    // Show default recommendations based on preferred genre
    showRecommendations(userPreferences.preferredGenre);
});

// Search functionality
document.getElementById('searchBtn').addEventListener('click', function() {
    console.log('Search button clicked');
    const searchTerm = document.getElementById('searchInput').value.toLowerCase().trim();
    console.log('Search term:', searchTerm);
    if (searchTerm) {
        console.log('Calling searchMovies with:', searchTerm);
        searchMovies(searchTerm);
    } else {
        console.log('Empty search term, showing recommendations for preferred genre');
        if (userPreferences && userPreferences.preferredGenre) {
            showRecommendations(userPreferences.preferredGenre);
        } else {
            // If no user preferences, show some default movies
            console.log('No user preferences, showing default movies');
            displayMovies(movieDatabase.action || []);
        }
    }
});

// Search on Enter key
document.getElementById('searchInput').addEventListener('keypress', function(e) {
    console.log('Search input keypress:', e.key);
    if (e.key === 'Enter') {
        console.log('Enter key pressed, clicking search button');
        document.getElementById('searchBtn').click();
    }
});

function showRecommendations(genre) {
    currentMovies = movieDatabase[genre] || [];
    displayMovies(currentMovies);
}

async function searchMovies(searchTerm) {
    try {
        console.log('Searching for:', searchTerm);
        
        // Get search results from Flask backend
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ movie_name: searchTerm })
        });
        
        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Search response:', data);
        console.log('Recommendations length:', data.recommendations ? data.recommendations.length : 'undefined');
        
        if (data.error) {
            console.error('Search error:', data.error);
            displayMovies([]);
            return;
        }
        
        if (!data.recommendations || data.recommendations.length === 0) {
            console.log('No recommendations found');
            displayMovies([]);
            return;
        }
        
        // Format the search results for display
        const movies = data.recommendations.map(movie => ({
            title: movie.title,
            year: new Date().getFullYear(), // We don't have year in the dataset
            genre: movie.description || 'Action, Adventure',
            rating: movie.vote_average || movie.score * 10,
            plot: movie.description || 'Movie description not available',
            poster: 'https://via.placeholder.com/300x200?text=Movie+Poster'
        }));
        
        console.log('Formatted movies:', movies);
        console.log('Movies array length:', movies.length);
        currentMovies = movies;
        displayMovies(movies);
        
    } catch (error) {
        console.error('Error searching movies:', error);
        displayMovies([]);
    }
}

function displayMovies(movies) {
    console.log('displayMovies called with:', movies);
    console.log('Movies length:', movies.length);
    
    const moviesGrid = document.getElementById('moviesGrid');
    console.log('moviesGrid element:', moviesGrid);
    
    if (!moviesGrid) {
        console.error('moviesGrid element not found!');
        return;
    }
    
    if (movies.length === 0) {
        console.log('No movies to display, showing "No movies found" message');
        moviesGrid.innerHTML = '<p style="text-align: center; color: #666; font-size: 1.2rem;">No movies found. Try a different search term.</p>';
        return;
    }
    
    console.log('Displaying', movies.length, 'movies');
    moviesGrid.innerHTML = movies.map(movie => `
        <div class="movie-card">
            <img src="${movie.poster}" alt="${movie.title}" class="movie-poster" onerror="this.src='https://via.placeholder.com/300x200?text=Movie+Poster'">
            <div class="movie-info">
                <div class="movie-title">${movie.title}</div>
                <div class="movie-year">${movie.year}</div>
                <div class="movie-genre">${movie.genre}</div>
                <div class="movie-rating">
                    <span>IMDb Rating:</span>
                    <span class="imdb-rating">${movie.rating}/10</span>
                </div>
                <div class="movie-plot">${movie.plot}</div>
            </div>
        </div>
    `).join('');
}
