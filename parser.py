import requests
from bs4 import BeautifulSoup

URL = 'https://www.imdb.com/title/tt9140554/reviews?ref_=tt_urv'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

response = requests.get(URL, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

reviews = soup.find_all('div', class_='lister-item-content')

for review in reviews:
    # Get the rating
    rating = int(review.find('span', class_='rating-other-user-rating').span.text)
    if rating >= 7:
        sentiment = 'positive'
    elif rating < 5:
        sentiment = 'negative'
    else:
        continue  # Skip neutral reviews

    # Get the title and content of the review
    title = review.find('a', class_='title').text.strip()
    content = review.find('div', class_='text show-more__control').text.strip()

    print(f"Title: {title}")
    print(f"Rating: {rating} ({sentiment})")
    print(f"Review:\n{content}\n")
    print('-' * 50)
