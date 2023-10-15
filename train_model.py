path_to_data = 'News.csv'

#1.Prepare the data
prepared_data = dp.prepare_data(path_to_data)

#2.Create train -test split
train_test_data,vectorizer = dp.create_train_test_data(prepared_data['text'],
                                                      prepared_data['class'],
                                                      0.33,2023)

#3.Runtraining
model = mt.run_model_training(train_test_data['X_train'],train_test_data['X_test'],
                             train_test_data['y_train'],train_test_data['y_test'])



#save the trained model and vectorizer
joblib.dump(model,'my_fake_news.pkl')
joblib.dump(cv,'my_vectorizer_news.pkl')

