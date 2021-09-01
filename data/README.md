This is the dataset containing tweets for POI type prediction.

If you use this data please cite the papers: 
- D. Sánchez Villegas, D. Preoţiuc-Pietro and N. Aletras (2020). Point-of-Interest Type Inference from Social Media Text. In AACL. 
  
  [paper](https://aclanthology.org/2020.aacl-main.80/) | [data](https://archive.org/details/poi-data)
- D. Sánchez Villegas and N. Aletras (2021).Point-of-Interest Type Prediction using Text and Images. In EMNLP. 

Each row represents one tweet, containing {"tweet_id","split","has_image","category_name"}.
- tweet_id: the id of the tweet
- split: train/dev/test
- has_image: True/False indicates that the tweet has image content and was utilized as input for the models
- category_name: the POI category

The tweet IDs can be used to retrieve the original tweet using the Twitter API, alongside any other information you may require. 
Note that some of the tweets may have been deleted or set to private.
