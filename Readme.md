The gradio webapp relies on three pickle files that are created in a jupyter notebook. The gradio webapp file is as follows: "gradio_app.py".
All the details about the above-mentioned jupyter notebook and the pickle files are reported below. 

Jupyter notebook: 
- The jupyter notebook named "training_notebook.ipynb" trains a XGboost model based on TfIdfVecotrizer, 
which is applied on a dataset on hotel reviews and hotel ratings. 
- The dataset used in this notebook comes from Kaggle: https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe (see the CSV file "Hotel_Reviews.csv")

The 3 pickle files: 
- The jupyter notebook stores 3 pickles. The 3 pickle files are named "vectorizer.pkl"; "feature_names.pkl"; "xgboost_model.pkl".
- Finally, the 3 pickle files are used by the "gradio_app.py" file to predict the rating for a new unseen review. 
