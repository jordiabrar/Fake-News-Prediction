# Fake News Prediction

A Python project that uses machine learning to predict whether a given news article is real or fake. This project employs the Naive Bayes algorithm and TF-IDF vectorization for text processing.

## Features
- Preprocess and clean news data.
- Train a Naive Bayes classifier to distinguish between real and fake news.
- Evaluate model performance using metrics like accuracy, confusion matrix, and classification report.
- Test the model with custom news input.

---

## Requirements
Install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

---

## Dataset
The dataset should be a CSV file with the following structure:

| text                                              | label |
|---------------------------------------------------|-------|
| Breaking news: Scientists discover water on Mars! | REAL  |
| Click here to win a free iPhone now!              | FAKE  |
| New study shows coffee boosts productivity by 50%!| REAL  |

- `text`: Contains the news content.
- `label`: Indicates whether the news is `REAL` or `FAKE`.

Save the dataset as `fake_or_real_news.csv` in the project directory.

---

## File Structure
```
Fake-News-Prediction/
├── fake_news_prediction.py  # Main Python script
├── fake_or_real_news.csv    # Dataset file
├── README.md                # Project documentation
```

---

## How to Run
1. Clone this repository or download the files.
2. Ensure the dataset file (`fake_or_real_news.csv`) is in the project directory.
3. Run the Python script:

```bash
python fake_news_prediction.py
```

4. The script will output the model's accuracy, confusion matrix, and classification report. You can also test it with custom news input.

---

## Example Usage
### Code Output
```
Accuracy: 0.92
Confusion Matrix:
[[50  5]
 [ 4 41]]
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.91      0.92        55
           1       0.89      0.91      0.90        45

    accuracy                           0.92       100
   macro avg       0.91      0.92      0.91       100
weighted avg       0.92      0.92      0.92       100
```

### Test Custom Input
```python
custom_news = "Breaking news: Scientists discover water on Mars!"
print(f"Prediction: {predict_news(custom_news)}")
```
Output:
```
Prediction: REAL
```

---

## Future Improvements
- Enhance text preprocessing (e.g., remove stopwords, stemming).
- Explore other machine learning algorithms (e.g., Logistic Regression, SVM).
- Deploy the model as a web app using Flask or Django.