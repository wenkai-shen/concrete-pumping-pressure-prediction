Pump pressure model.py ：The process of training the model
example input output files.xlsx ： Examples of input and output parameters for the model dataset
predict pump pressure.py ： Predict this script to obtain the prediction result
pump_pressure_model.pkl ： The model file trained by myself
scaler.pkl ： It is used to standardize the input data, ensuring that the new data maintains the same scale as the training data
X_train_scaled.npy ： The training set is a standardized feature matrix used for cluster matching
dbscan_labels.npy ： The labels obtained after DBSCAN clustering on the training set are used to match categories for new samples
