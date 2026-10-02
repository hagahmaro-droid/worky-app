HOTEL BOOKING CANCELLATION PREDICTOR
===================================

Files:
- app.py                         Streamlit UI
- train_model.py                training/export script
- hotel_cancellation_model.pkl  trained XGBoost + preprocessing pipeline
- hotel_bookings.csv            dataset

RUN LOCALLY
-----------
1) Open a terminal in this folder.
2) Install dependencies:
   pip install -r requirements.txt
3) Run:
   streamlit run app.py

If you change the dataset or the training code, retrain first:
   python train_model.py

The app automatically loads the saved model and sends user-entered booking data through
THE SAME preprocessing pipeline before prediction.
