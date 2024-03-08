# Automated Sentiment Analysis of Movie Reviews

### Project Overview
This project implements a sentiment analysis tool using Logistic Regression to classify movie reviews as positive or negative. It was developed during an exchange program at the Beijing Institute of Technology as part of the "Introduction to Big Data Analysis" course.

### Motivation
The project addresses the challenge of information overload in the digital age, where users struggle to navigate through vast amounts of user-generated content, such as online reviews. Manually analyzing reviews to understand the overall sentiment is time-consuming and inefficient.

### Methodology
A Logistic Regression model was implemented to categorize movie reviews as positive or negative. This approach was chosen due to its:
- **Simplicity:** Facilitating understanding and interpretation of the model.
- **Efficiency:** Enabling efficient training and prediction.
- **Interpretability:** Offering insights into the features most influential in sentiment determination.

The project encompassed the following stages:
1) **Data Collection and Preprocessing**
    - Split the dataset into training and validation sets.
    - Employed the "IMDb Movie Review" dataset from Kaggle.
    - Augmented the dataset with reviews scraped from the IMDb website using web scraping techniques.
    - Processed and tokenized reviews for feature engineering.
2) **Model Training and Evaluation**
    - Split the dataset into training and validation sets.
    - Trained the Logistic Regression model on the training set, incorporating regularization to prevent overfitting.
    - Evaluated model performance on the validation set using metrics like accuracy, precision, recall, and F1-score.
3) **Development of a User Interface**
    - Created a Python Flask application for interactive sentiment analysis.
    - The application provides an interface for users to input text and receive real-time sentiment predictions with confidence scores.
    - Visualizations using emojis enhance user understanding of the model's output.
      ![Без заголовка](https://github.com/kidavspb/ya2cn/assets/84584461/fd39aec2-37dd-40f0-8b1a-792768254b94)



### Results
The Logistic Regression model achieved an accuracy of 88% in classifying movie reviews. Feature engineering techniques played a crucial role in model performance, highlighting the importance of specific words in sentiment determination. The model also demonstrated robustness when evaluated on the augmented data from web scraping.
|              | precision | recall | f1-score | support |
|-------------:|-----------|--------|----------|---------|
|     Negative | 0.90      | 0.86   | 0.88     | 1018    |
|     Positive | 0.86      | 0.90   | 0.88     | 982     |
|     accuracy |           |        | 0.88     | 2000    |
|    macro avg | 0.88      | 0.88   | 0.88     | 2000    |
| weighted avg | 0.88      | 0.88   | 0.88     | 2000    |


### Future Work
This project lays the groundwork for further exploration in sentiment analysis:
- **Summarize pros and cons** from review, inspired by platforms like Yandex.Market.
- **Perform aspect-based sentiment analysis** to understand sentiment towards specific aspects of movies (acting, direction, etc.).
- **Explore more advanced techniques** like ensemble methods for potentially higher accuracy.

### Conclusion
This project successfully developed a sentiment analysis tool using Logistic Regression, demonstrating its effectiveness in classifying movie reviews.  The research underscores the potential of sentiment analysis to enhance user experience in navigating the vast amount of user-generated content within the digital realm. The project's findings contribute to the field of Big Data Analysis and lay a foundation for further advancements in sentiment analysis techniques.
