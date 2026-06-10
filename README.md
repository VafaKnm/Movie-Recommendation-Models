# Movie Recommendation Models

A movie recommendation project built with both **Content-Based Recommendation** and **Collaborative Filtering** approaches.

This project was originally developed in a Kaggle Notebook environment and demonstrates how different recommendation system techniques can be used to suggest movies to users.

---

## Overview

Recommendation systems help users discover items they may like.

In this project, the target item is a movie.

The system can recommend movies using two main strategies:

1. **Content-Based Recommendation**
2. **Collaborative Filtering**

These two approaches solve the recommendation problem in different ways.

```text id="gzwtl0"
Content-Based:
    Recommend movies similar to a selected movie

Collaborative Filtering:
    Recommend movies based on user-item interaction patterns
```

---

## What This Project Does

This project demonstrates how to:

* Load movie-related datasets
* Explore movie metadata
* Build a content-based movie recommender
* Build a collaborative filtering recommender
* Compare different recommendation strategies
* Generate movie recommendations for a selected movie or user
* Understand the strengths and weaknesses of each approach

---

## Recommendation Approaches

## 1. Content-Based Recommendation

Content-based recommendation uses movie features to find similar movies.

Possible movie features include:

```text id="pt9lhk"
title
genres
overview
keywords
cast
director
production company
release year
language
runtime
```

The main idea is:

```text id="my56st"
If a user likes Movie A,
recommend movies that are similar to Movie A.
```

Example:

```text id="ljdu50"
Input movie:
    The Dark Knight

Recommended movies:
    Batman Begins
    The Dark Knight Rises
    Man of Steel
    Watchmen
```

Content-based systems are useful when we have rich metadata about movies.

---

## 2. Collaborative Filtering

Collaborative filtering uses user behavior or ratings.

The main idea is:

```text id="477ptp"
Users who liked similar movies in the past
may like similar movies in the future.
```

Collaborative filtering can be user-based or item-based.

### User-Based Collaborative Filtering

Find users with similar taste and recommend movies they liked.

```text id="3755wj"
User A likes movies 1, 2, 3
User B likes movies 1, 2, 4

User A may like movie 4
```

### Item-Based Collaborative Filtering

Find movies that receive similar rating patterns from users.

```text id="28qhhy"
Users who liked Movie A also liked Movie B
```

Collaborative filtering is powerful when user-rating data is available.

---

## Content-Based Pipeline

A typical content-based recommendation pipeline looks like this:

```text id="4oc3j8"
Movie Metadata
      ↓
Text Feature Cleaning
      ↓
Feature Combination
      ↓
Vectorization
      ↓
Similarity Matrix
      ↓
Input Movie
      ↓
Top-N Similar Movies
```

Common vectorization methods:

```text id="cns60e"
CountVectorizer
TF-IDF Vectorizer
Text embeddings
```

Common similarity metrics:

```text id="8g5hmu"
Cosine Similarity
Euclidean Distance
Jaccard Similarity
```

---

## Collaborative Filtering Pipeline

A typical collaborative filtering pipeline looks like this:

```text id="cgssfu"
User-Movie Ratings
      ↓
User-Item Matrix
      ↓
Similarity Calculation or Matrix Factorization
      ↓
Predict Missing Ratings
      ↓
Recommend Top-N Movies
```

Possible collaborative filtering methods:

```text id="t9ft1m"
User-user similarity
Item-item similarity
KNN collaborative filtering
Matrix factorization
SVD
NMF
Neural collaborative filtering
```

---

## Input and Output

### Content-Based Input

The input can be a movie title.

Example:

```text id="xi0g5l"
Input movie: Inception
```

### Content-Based Output

The output is a list of similar movies.

Example:

```text id="ao3cfs"
Recommended movies:
1. Interstellar
2. The Prestige
3. Memento
4. Shutter Island
5. The Matrix
```

---

### Collaborative Filtering Input

The input can be a user ID.

Example:

```text id="5sh6x1"
Input user: user_123
```

### Collaborative Filtering Output

The output is a list of movies that the user may like.

Example:

```text id="dbifuj"
Recommended movies:
1. Movie A
2. Movie B
3. Movie C
```

---

## Dataset

The project uses movie recommendation data in a Kaggle environment.

A typical movie recommendation dataset may contain:

### Movie Metadata

```text id="n8d2mq"
movie_id
title
genres
overview
keywords
cast
crew
release_date
language
popularity
vote_average
vote_count
```

### User Ratings

```text id="26o6af"
user_id
movie_id
rating
timestamp
```

The exact dataset version depends on the Kaggle notebook used in the project.

---

## Feature Engineering

For content-based recommendation, several text features can be combined into one field.

Example:

```text id="z8ykqg"
genres + keywords + overview + cast + director
```

This combined text representation can then be vectorized.

Example:

```python id="j0rjd1"
df["combined_features"] = (
    df["genres"].fillna("") + " " +
    df["keywords"].fillna("") + " " +
    df["overview"].fillna("") + " " +
    df["cast"].fillna("") + " " +
    df["director"].fillna("")
)
```

---

## Cosine Similarity

Cosine similarity is commonly used for content-based recommendation.

It measures how similar two movie vectors are.

```python id="rnra5s"
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(movie_vectors)
```

Conceptually:

```text id="mznwu0"
similarity_matrix[i][j] = similarity between movie i and movie j
```

---

## Example Content-Based Recommender

```python id="s4b4zx"
def recommend_similar_movies(movie_title, movies_df, similarity_matrix, top_n=10):
    """
    Recommend movies similar to the selected movie.
    """
    movie_index = movies_df[movies_df["title"] == movie_title].index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    movie_indices = [
        index for index, score in similarity_scores[1:top_n + 1]
    ]

    return movies_df.iloc[movie_indices][["title"]]
```

---

## Example Collaborative Filtering Logic

A simple item-based collaborative filtering approach can use a user-movie rating matrix.

```python id="7mc64d"
user_movie_matrix = ratings_df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
)
```

Then similarity can be calculated between movies or users.

```python id="t6gxys"
from sklearn.metrics.pairwise import cosine_similarity

movie_similarity = cosine_similarity(
    user_movie_matrix.fillna(0).T
)
```

---

## Repository Structure

Current repository structure:

```text id="4zub4j"
Movie-Recommendation-Models/
│
├── README.md
└── content-based-collaborative-filtering-for-movies.ipynb
```

Suggested future structure:

```text id="effvx5"
Movie-Recommendation-Models/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── content-based-collaborative-filtering-for-movies.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── content_based.py
│   ├── collaborative_filtering.py
│   ├── evaluate.py
│   └── app.py
├── models/
│   ├── vectorizer.pkl
│   ├── content_similarity_matrix.pkl
│   └── collaborative_model.pkl
├── assets/
│   ├── recommendation_example.png
│   ├── rating_distribution.png
│   └── model_comparison.png
└── examples/
    └── sample_recommendations.md
```

---

## Installation

Clone the repository:

```bash id="wq032t"
git clone https://github.com/VafaKnm/Movie-Recommendation-Models.git
cd Movie-Recommendation-Models
```

Create a virtual environment:

```bash id="pwxqdt"
python -m venv .venv
source .venv/bin/activate
```

Install common dependencies:

```bash id="hfwf1u"
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash id="t6a9js"
jupyter notebook
```

Open:

```text id="fwirfz"
content-based-collaborative-filtering-for-movies.ipynb
```

---

## Suggested `requirements.txt`

```txt id="974v3n"
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

If matrix factorization or recommender libraries are added later:

```txt id="gzvj9h"
scipy
surprise
implicit
lightfm
faiss-cpu
```

If a demo is added later:

```txt id="t8iu1u"
streamlit
gradio
fastapi
uvicorn
```

---

## Evaluation

The current README does not include formal recommendation metrics.

Recommendation systems can be evaluated in different ways depending on the available data.

### Content-Based Evaluation

For content-based recommendation, useful checks include:

| Metric / Check     | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| Genre Match@K      | How many recommended movies share genres with the input movie |
| Similarity Score@K | Average similarity of top-K recommendations                   |
| Diversity@K        | How diverse the recommended movies are                        |
| Manual Review      | Human inspection of recommendation quality                    |

### Collaborative Filtering Evaluation

If user ratings are available, useful metrics include:

| Metric      | Description                                         |
| ----------- | --------------------------------------------------- |
| RMSE        | Error in predicted ratings                          |
| MAE         | Average absolute rating error                       |
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K    | Fraction of relevant movies retrieved               |
| MAP@K       | Ranking quality                                     |
| NDCG@K      | Ranking quality with graded relevance               |
| Hit Rate@K  | Whether at least one relevant item appears in top K |

---

## Suggested Result Table

After evaluation, the README can be updated with:

| Model                   | Metric         | Score |
| ----------------------- | -------------- | ----: |
| Content-Based           | Genre Match@10 |   TBD |
| Content-Based           | Diversity@10   |   TBD |
| Collaborative Filtering | RMSE           |   TBD |
| Collaborative Filtering | MAE            |   TBD |
| Collaborative Filtering | Precision@10   |   TBD |
| Collaborative Filtering | NDCG@10        |   TBD |

---

## Suggested Improvements

### 1. Add Real Recommendation Examples

The README should include real examples from the notebook.

Example:

```text id="u132hw"
Input movie:
    Toy Story

Recommended movies:
    1. A Bug's Life
    2. Monsters, Inc.
    3. Finding Nemo
    4. Shrek
    5. Ice Age
```

This makes the project much easier to understand.

---

### 2. Add Evaluation Metrics

Add metrics for both recommendation types.

Recommended first metrics:

```text id="4tvxqn"
Genre Match@10
Diversity@10
RMSE
MAE
Precision@10
NDCG@10
```

---

### 3. Add Matrix Factorization

Collaborative filtering can be improved with matrix factorization.

Recommended methods:

```text id="o5rmp5"
SVD
NMF
ALS
LightFM
Neural Collaborative Filtering
```

---

### 4. Add Hybrid Recommendation

A stronger recommender can combine content-based and collaborative filtering.

Conceptually:

```text id="h9fonz"
final_score =
    alpha * content_similarity_score
    + beta * collaborative_score
```

This can improve recommendations because it uses both movie metadata and user behavior.

---

### 5. Add User Profile Recommendation

For a user, the model can build a user profile based on movies they liked.

Example:

```text id="nezk06"
User liked:
    Inception
    Interstellar
    The Matrix

Recommend:
    similar science-fiction / thriller movies
```

---

### 6. Add Diversity Control

A recommender should avoid returning nearly identical movies.

Possible diversity rules:

```text id="ys5xcl"
limit same franchise
limit same director
limit same genre repetition
include diverse genres
```

---

### 7. Add Search and Fuzzy Matching

Movie title input can be difficult because users may type incomplete names.

Useful improvements:

```text id="dbavv0"
case-insensitive search
partial title search
fuzzy matching
autocomplete
```

Example:

```text id="rf84h3"
"dark knight" → "The Dark Knight"
```

---

### 8. Add Web Demo

A simple demo would make the project easier to test.

Recommended tools:

* Streamlit
* Gradio
* FastAPI

Example UI:

```text id="xots75"
Search movie title
      ↓
Select recommendation method
      ↓
Choose top-N
      ↓
Generate recommendations
      ↓
Show movie titles, genres, similarity scores
```

---

## Limitations

The current project has several limitations:

* It is notebook-based and not yet structured as reusable Python code.
* The README does not currently include real recommendation examples.
* The README does not include formal evaluation metrics.
* Content-based recommendation may recommend movies that are too similar.
* Collaborative filtering needs enough user-rating data.
* Cold-start users are difficult for collaborative filtering.
* Cold-start movies are difficult if metadata is incomplete.
* Popularity bias may affect recommendation quality.
* Movie taste is subjective and hard to evaluate with only offline metrics.

---

## Use Cases

This project can be useful for:

* Learning recommendation systems
* Understanding content-based filtering
* Understanding collaborative filtering
* Practicing cosine similarity
* Working with movie metadata
* Working with user-rating data
* Building a recommender baseline
* Creating a movie recommendation demo
* Comparing different recommendation approaches

---

## Possible Future Work

Recommended future work:

* Add `requirements.txt`
* Add clean Python scripts
* Add saved vectorizer and similarity matrix
* Add real sample recommendations
* Add evaluation metrics
* Add matrix factorization model
* Add hybrid recommendation
* Add fuzzy title search
* Add diversity control
* Add popularity bias analysis
* Add Streamlit or Gradio demo
* Add FastAPI endpoint
* Add Dockerfile

---

## Conclusion

This repository demonstrates movie recommendation using both content-based recommendation and collaborative filtering.

The project is useful for learning:

```text id="ney19z"
movie recommendation systems
content-based filtering
collaborative filtering
similarity calculation
user-item matrices
top-N recommendation
Kaggle-based recommender workflow
```

With real recommendation examples, evaluation metrics, a reusable code structure, and a simple demo interface, this repository can become a much stronger recommendation system portfolio project.
