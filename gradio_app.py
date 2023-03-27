import pandas as pd
import pickle
import gradio as gr

with open('xgboost_model.pkl', 'rb') as f_new:
    reloaded_model = pickle.load(f_new)

with open('vectorizer.pkl', 'rb') as f_new:
    vectorizer = pickle.load(f_new)

with open('feature_names.pkl', 'rb') as f_new:
    feature_names = pickle.load(f_new)

def return_numerical_prediction(new_review):
    # Transform the preprocessed review into a feature vector
    new_review_vector = vectorizer.transform([new_review])

    # Create a new dataframe with the feature vector
    new_review_df = pd.DataFrame(new_review_vector.toarray(), columns=feature_names)

    # Predict the binary score of the new review
    new_review_score = reloaded_model.predict(new_review_df)[0]
    return new_review_score

def predict_sentiment(review):
    new_review_score = return_numerical_prediction(review)
    if new_review_score == 1:
        return "Positive review"
    else:
        return "Negative review"

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.inputs.Textbox(placeholder="Enter your review here"),
    outputs=gr.outputs.Textbox(label="Sentiment prediction")
)

iface.launch(share=True)
