import json
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load JSON data
with open('train.json', 'r') as file:
    data = json.load(file)

# Extract reviews and sentiments
reviews = [item['reviews'] for item in data]
sentiments = [item['sentiments'] for item in data]

# Initialize WordNet Lemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()

# Define a function for text preprocessing with lemmatization and tokenization
def preprocess_text(text):
    tokens = word_tokenize(text.translate(str.maketrans('', '', string.punctuation)))
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Create list of preprocessed reviews
preprocessed_reviews = [' '.join(preprocess_text(review)) for review in reviews]

# Create list of dictionaries containing original reviews, preprocessed reviews, and sentiments
processed_data = []
for i in range(len(reviews)):
    processed_data.append({
        'original_reviews': reviews[i],
        'preprocessed_reviews': preprocessed_reviews[i],
        'sentiments': sentiments[i]
    })

# Write processed data to a new JSON file
with open('preprocessed.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("Data preprocessed and saved to 'preprocessed.json'")
