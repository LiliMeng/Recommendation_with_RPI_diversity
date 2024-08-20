# Recommendation_with_RPI_diversity

## With RPI

Integrating item-item collaborative filtering with revenue-related ranking in a recommendation system involves combining the similarity-based recommendations of item-item collaborative filtering with a revenue optimization strategy. This approach ensures that the recommended items are both relevant to the user (based on past interactions) and have a high potential to generate revenue.

### 1. **Example Dataset**

Let's start by defining an example dataset that includes both user-item interactions and revenue-related information.

#### User-Item Interaction Data
This data represents users interacting with various items. The interaction could be clicks, views, or ratings.

| User | Item A | Item B | Item C | Item D |
|------|--------|--------|--------|--------|
|  U1  |   5    |   3    |   0    |   1    |
|  U2  |   4    |   0    |   2    |   0    |
|  U3  |   0    |   3    |   4    |   5    |
|  U4  |   3    |   4    |   0    |   2    |

#### Revenue Data
This data represents the revenue-related information for each item.

| Item | Price | Conversion Rate | CTR  | Historical Revenue |
|------|-------|-----------------|------|--------------------|
|  A   |  100  |      0.05        | 0.10 |       5000         |
|  B   |  150  |      0.03        | 0.15 |       4500         |
|  C   |  200  |      0.04        | 0.07 |       8000         |
|  D   |  50   |      0.10        | 0.05 |       3000         |

### 2. **Step-by-Step Integration**

#### a. **Compute Item-Item Similarity**

First, we compute the similarity between items based on the user-item interaction data using cosine similarity.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example user-item interaction matrix (as a NumPy array)
user_item_matrix = np.array([
    [5, 3, 0, 1],
    [4, 0, 2, 0],
    [0, 3, 4, 5],
    [3, 4, 0, 2]
])

# Transpose the matrix to get item vectors
item_item_matrix = user_item_matrix.T

# Compute cosine similarity between items
item_similarity_matrix = cosine_similarity(item_item_matrix)

# Display the item-item similarity matrix
print("Item-Item Similarity Matrix:")
print(item_similarity_matrix)
```

#### b. **Calculate Revenue Potential (RPI)**

Next, we calculate the Revenue Per Impression (RPI) for each item based on its price, conversion rate, and CTR.

```python
# Example revenue data
items = [
    {"id": "A", "price": 100, "conversion_rate": 0.05, "ctr": 0.10},
    {"id": "B", "price": 150, "conversion_rate": 0.03, "ctr": 0.15},
    {"id": "C", "price": 200, "conversion_rate": 0.04, "ctr": 0.07},
    {"id": "D", "price": 50, "conversion_rate": 0.10, "ctr": 0.05},
]

# Calculate RPI for each item
for item in items:
    item["rpi"] = item["ctr"] * item["conversion_rate"] * item["price"]

# Display RPI values
for item in items:
    print(f"Item {item['id']} - RPI: {item['rpi']}")
```

#### c. **Generate Initial Recommendations Using Item-Item Collaborative Filtering**

Based on the item-item similarity matrix, generate a list of similar items for a given item the user has interacted with.

```python
def get_similar_items(item_id, item_similarity_matrix, item_index_map, top_n=3):
    # Get the index of the item in the matrix
    item_index = item_index_map[item_id]
    # Get similarity scores for the item
    similarity_scores = item_similarity_matrix[item_index]
    # Sort items by similarity score
    similar_items = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)
    # Get the top N similar items, excluding the item itself
    similar_items = [(index, score) for index, score in similar_items if index != item_index][:top_n]
    return similar_items

# Create a mapping from item IDs to indices
item_index_map = {"A": 0, "B": 1, "C": 2, "D": 3}

# Get similar items for Item A
similar_items = get_similar_items("A", item_similarity_matrix, item_index_map)
print("Similar Items to A:")
print(similar_items)
```

#### d. **Adjust Recommendations Based on Revenue Potential**

Adjust the ranking of the recommended items by incorporating their RPI values into the ranking score.

```python
def combined_score(similarity_score, rpi, alpha=0.7, beta=0.3):
    return alpha * similarity_score + beta * rpi

# Combine similarity score with RPI for ranking
ranked_similar_items = []
for index, similarity_score in similar_items:
    item_id = list(item_index_map.keys())[index]
    item_rpi = next(item["rpi"] for item in items if item["id"] == item_id)
    score = combined_score(similarity_score, item_rpi)
    ranked_similar_items.append((item_id, score))

# Sort by combined score
ranked_similar_items = sorted(ranked_similar_items, key=lambda x: x[1], reverse=True)

# Display the final ranked list of recommended items
print("Ranked Recommendations for A:")
print(ranked_similar_items)
```

### 3. **Example Output**

Suppose the cosine similarity matrix and RPI values calculated as follows:

- **Item-Item Similarity Matrix** (after computation):
  ```
  [[1. , 0.7962, 0.3586, 0.5726],
   [0.7962, 1. , 0.5557, 0.4803],
   [0.3586, 0.5557, 1. , 0.8054],
   [0.5726, 0.4803, 0.8054, 1. ]]
  ```

- **RPI Values**:
  ```
  Item A - RPI: 0.5
  Item B - RPI: 0.675
  Item C - RPI: 0.56
  Item D - RPI: 0.25
  ```

- **Similar Items to A (Before Revenue Adjustment)**:
  ```
  [(1, 0.7962), (3, 0.5726), (2, 0.3586)]
  ```

- **Ranked Recommendations for A (After Revenue Adjustment)**:
  ```
  [('B', 0.76884), ('C', 0.43202), ('D', 0.48182)]
  ```

In this example, the integration of RPI with item similarity led to the item "B" being ranked higher than others, even if it wasn't the most similar to "A" in terms of user interaction alone, because of its higher revenue potential.

### Conclusion

This approach allows you to combine the strengths of item-item collaborative filtering with revenue-based ranking, ensuring that the recommendations are both relevant and aligned with business goals like revenue optimization. The key is to balance similarity (relevance) with revenue potential, which can be fine-tuned using parameters like `alpha` and `beta` in the combined score formula.

## With diversity
Adding diversity to the recommendation system that integrates item-item collaborative filtering with revenue-based ranking involves introducing mechanisms that ensure the recommended items are not only relevant and profitable but also diverse in terms of categories, brands, or other attributes. Below are steps to incorporate diversity into the recommendation system.

### 1. **Introduce a Diversity Factor**

One way to introduce diversity is to modify the ranking score by incorporating a diversity penalty or adjustment. This can be done by penalizing items that are too similar to each other or by rewarding items that are different from those already recommended.

#### a. **Diversity Score Calculation**
   - **Diversity Score**: Calculate a diversity score that represents how different an item is from the items already in the recommendation list.
   - **Similarity Penalty**: Penalize items that are too similar to those already recommended.

#### b. **Combined Score with Diversity**
  <img width="685" alt="Screenshot 2024-08-20 at 11 16 22 AM" src="https://github.com/user-attachments/assets/b32e5bea-044d-434d-b046-00a0572428ce">


### 2. **Implementing Diversity in Code**

Here’s how you can modify the earlier code to include a diversity factor:

```python
def calculate_diversity_penalty(candidate_item, selected_items, item_similarity_matrix, item_index_map):
    # Calculate the penalty based on how similar the candidate item is to already selected items
    penalty = 0
    for selected_item in selected_items:
        selected_index = item_index_map[selected_item]
        candidate_index = item_index_map[candidate_item]
        penalty += item_similarity_matrix[selected_index][candidate_index]
    return penalty / len(selected_items) if selected_items else 0

def combined_score_with_diversity(similarity_score, rpi, diversity_penalty, alpha=0.6, beta=0.3, gamma=0.1):
    return alpha * similarity_score + beta * rpi - gamma * diversity_penalty

# Example: Integrating diversity into the recommendation process
def get_diverse_recommendations(item_id, item_similarity_matrix, item_index_map, items, top_n=3):
    similar_items = get_similar_items(item_id, item_similarity_matrix, item_index_map)
    
    selected_items = []  # This will store the items already recommended
    ranked_similar_items = []

    for index, similarity_score in similar_items:
        candidate_item_id = list(item_index_map.keys())[index]
        candidate_rpi = next(item["rpi"] for item in items if item["id"] == candidate_item_id)
        diversity_penalty = calculate_diversity_penalty(candidate_item_id, selected_items, item_similarity_matrix, item_index_map)
        score = combined_score_with_diversity(similarity_score, candidate_rpi, diversity_penalty)
        ranked_similar_items.append((candidate_item_id, score))

    # Sort by the combined score
    ranked_similar_items = sorted(ranked_similar_items, key=lambda x: x[1], reverse=True)

    # Select the top N diverse recommendations
    selected_items = [item_id for item_id, _ in ranked_similar_items[:top_n]]
    
    return selected_items

# Generate diverse recommendations for Item A
diverse_recommendations = get_diverse_recommendations("A", item_similarity_matrix, item_index_map, items)
print("Diverse Recommendations for A:")
print(diverse_recommendations)
```

### 3. **Category-Based Diversity**

Another approach to ensuring diversity is to limit the number of items from the same category in the top N recommendations. For instance, you can implement a rule that no more than one or two items from the same category should appear in the top N recommendations.

#### a. **Category Filtering**
   - **Category Constraint**: After generating recommendations, filter them based on categories to ensure no category dominates the recommendation list.

#### b. **Category-Based Re-Ranking**
   - Re-rank the items to ensure a balanced representation of different categories.

```python
def get_category_diverse_recommendations(item_id, item_similarity_matrix, item_index_map, items, top_n=3):
    similar_items = get_similar_items(item_id, item_similarity_matrix, item_index_map)

    selected_items = []
    category_count = {}

    for index, similarity_score in similar_items:
        candidate_item_id = list(item_index_map.keys())[index]
        candidate_item = next(item for item in items if item["id"] == candidate_item_id)
        candidate_category = candidate_item.get("category", "default")

        # Apply category constraint
        if category_count.get(candidate_category, 0) < 1:  # Limit to 1 per category
            selected_items.append(candidate_item_id)
            category_count[candidate_category] = category_count.get(candidate_category, 0) + 1

        if len(selected_items) >= top_n:
            break

    return selected_items

# Example: Assume each item has a category
items = [
    {"id": "A", "price": 100, "conversion_rate": 0.05, "ctr": 0.10, "category": "Electronics"},
    {"id": "B", "price": 150, "conversion_rate": 0.03, "ctr": 0.15, "category": "Electronics"},
    {"id": "C", "price": 200, "conversion_rate": 0.04, "ctr": 0.07, "category": "Home"},
    {"id": "D", "price": 50, "conversion_rate": 0.10, "ctr": 0.05, "category": "Books"},
]

# Generate category-diverse recommendations for Item A
category_diverse_recommendations = get_category_diverse_recommendations("A", item_similarity_matrix, item_index_map, items)
print("Category-Diverse Recommendations for A:")
print(category_diverse_recommendations)
```

### 4. **Re-Ranking for Diversity After Initial Recommendation**

You can also implement a re-ranking step after generating the initial list of recommendations to enforce diversity:

```python
def re_rank_with_diversity(initial_recommendations, items, diversity_factor=0.3):
    """
    Re-rank items to enforce diversity by penalizing items that are too similar to already ranked ones.
    """
    ranked_items = []
    while initial_recommendations:
        item_id, initial_score = initial_recommendations.pop(0)
        diversity_penalty = calculate_diversity_penalty(item_id, ranked_items, item_similarity_matrix, item_index_map)
        final_score = initial_score - diversity_factor * diversity_penalty
        ranked_items.append((item_id, final_score))

    # Sort again by the final score
    ranked_items = sorted(ranked_items, key=lambda x: x[1], reverse=True)
    
    return [item_id for item_id, _ in ranked_items]

# Example: Generate initial recommendations and then re-rank for diversity
initial_recommendations = get_similar_items("A", item_similarity_matrix, item_index_map)
re_ranked_recommendations = re_rank_with_diversity(initial_recommendations, items)
print("Re-Ranked Recommendations with Diversity for A:")
print(re_ranked_recommendations)
```

### 5. **Multi-Objective Optimization**

To optimize for both relevance, revenue, and diversity, you can employ multi-objective optimization techniques where the recommendation model is trained to balance these objectives. This can be done by adjusting the weights in the loss function or by using methods like Pareto optimization.

### Conclusion

By introducing diversity into the recommendation system, you can enhance the user experience by offering a broader range of options, increasing the likelihood of users discovering new items, and reducing the risk of recommendation fatigue. The strategies mentioned above, such as diversity penalties, category constraints, and re-ranking, can be combined with item-item collaborative filtering and revenue-based ranking to create a more balanced and engaging recommendation system.

## With image and text features
Integrating additional item features such as text summaries, images, and other metadata into item-item collaborative filtering can enhance the recommendation system by leveraging more detailed information about each item. This approach is often referred to as **hybrid filtering**, where collaborative filtering is combined with content-based filtering. Below, I’ll explain how to integrate different types of item features into the item-item collaborative filtering process.

### 1. **Integrating Text Features (e.g., Text Summary)**

#### a. **Text Embedding**
- Convert text features (e.g., product descriptions, reviews, or summaries) into numerical vectors using techniques such as:
  - **TF-IDF**: A simple method that represents text as a weighted word frequency vector.
  - **Word2Vec / GloVe**: Pre-trained word embeddings that can represent text as a dense vector.
  - **BERT or Transformer-Based Models**: Advanced methods that produce contextual embeddings.

**Example: Using TF-IDF for Text Representation**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Example item descriptions
item_descriptions = {
    "A": "High-quality wireless headphones with noise-cancellation",
    "B": "Affordable smartphone with a great camera",
    "C": "Smart TV with 4K resolution and smart features",
    "D": "Compact digital camera with high resolution and zoom"
}

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(item_descriptions.values())

# Convert the sparse matrix to a dense format for easier manipulation
text_vectors = text_vectors.toarray()

# Create a mapping from item IDs to their text vectors
item_text_map = {item_id: text_vectors[i] for i, item_id in enumerate(item_descriptions.keys())}
```

#### b. **Compute Similarity Based on Text Features**

After obtaining the text vectors for each item, compute the similarity between items based on these text features:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between text vectors
text_similarity_matrix = cosine_similarity(text_vectors)

# Map the text similarity matrix to item IDs
item_text_similarity = {item_id: text_similarity_matrix[i] for i, item_id in enumerate(item_descriptions.keys())}

# Display the text similarity matrix
print("Text Similarity Matrix:")
print(text_similarity_matrix)
```

### 2. **Integrating Image Features**

#### a. **Image Embedding**
- Use pre-trained deep learning models (e.g., ResNet, VGG, or EfficientNet) to extract image features.
- The image is passed through a Convolutional Neural Network (CNN) to obtain a dense vector that represents the visual features of the item.

**Example: Using a Pre-Trained Model for Image Feature Extraction**

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Example function to preprocess image and extract features
def extract_image_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Example image paths for each item
item_images = {
    "A": "path_to_image_A.jpg",
    "B": "path_to_image_B.jpg",
    "C": "path_to_image_C.jpg",
    "D": "path_to_image_D.jpg"
}

# Extract features for each image
item_image_map = {item_id: extract_image_features(img_path) for item_id, img_path in item_images.items()}
```

#### b. **Compute Similarity Based on Image Features**

With the image features extracted, compute the similarity between items based on these image vectors:

```python
# Stack the image feature vectors into a matrix
image_vectors = np.stack(list(item_image_map.values()))

# Compute image similarity
image_similarity_matrix = cosine_similarity(image_vectors)

# Map the image similarity matrix to item IDs
item_image_similarity = {item_id: image_similarity_matrix[i] for i, item_id in enumerate(item_images.keys())}

# Display the image similarity matrix
print("Image Similarity Matrix:")
print(image_similarity_matrix)
```

### 3. **Combining Multiple Feature Similarities**

Now that you have similarity matrices based on collaborative filtering, text features, and image features, you can combine them to generate a more comprehensive similarity measure.

#### a. **Weighted Combination of Similarities**

You can combine the different similarity matrices using a weighted sum:

```python
# Example weights for each similarity type
alpha = 0.4  # Weight for collaborative filtering similarity
beta = 0.3   # Weight for text similarity
gamma = 0.3  # Weight for image similarity

# Combined similarity matrix
combined_similarity_matrix = (
    alpha * item_similarity_matrix + 
    beta * text_similarity_matrix + 
    gamma * image_similarity_matrix
)

# Map the combined similarity matrix to item IDs
item_combined_similarity = {item_id: combined_similarity_matrix[i] for i, item_id in enumerate(item_descriptions.keys())}

# Display the combined similarity matrix
print("Combined Similarity Matrix:")
print(combined_similarity_matrix)
```

#### b. **Generating Recommendations with Combined Similarity**

Use the combined similarity matrix to generate recommendations:

```python
def get_combined_similar_items(item_id, combined_similarity_matrix, item_index_map, top_n=3):
    item_index = item_index_map[item_id]
    similarity_scores = combined_similarity_matrix[item_index]
    similar_items = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)
    similar_items = [(index, score) for index, score in similar_items if index != item_index][:top_n]
    return similar_items

# Example: Get recommendations for Item A
combined_similar_items = get_combined_similar_items("A", combined_similarity_matrix, item_index_map)
print("Combined Similar Items to A:")
print(combined_similar_items)
```

### 4. **Re-Ranking Based on Revenue and Diversity**

Finally, you can adjust the recommendations by incorporating revenue and diversity metrics, as discussed earlier.

### Conclusion

By integrating text, image, and other item features into item-item collaborative filtering, you can significantly enhance the recommendation system's ability to understand the items and their similarities, leading to more accurate and diverse recommendations. The combined similarity approach allows you to leverage the strengths of different feature types, ensuring that the system is both comprehensive and tailored to the user's preferences.
