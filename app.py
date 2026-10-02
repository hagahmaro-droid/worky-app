import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent
bundle = joblib.load(BASE / 'hotel_cancellation_model.pkl')
model = bundle['model']
metrics = bundle['metrics']

st.set_page_config(page_title='Hotel Booking Predictor', page_icon='🏨', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.big-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem;}
.subtitle {font-size: 1.05rem; opacity: 0.75; margin-bottom: 1.5rem;}
.result {padding: 1.2rem; border-radius: 14px; border: 1px solid rgba(128,128,128,.25);}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="big-title">🏨 Hotel Booking Cancellation Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning application based on the Hotel Booking dataset.</div>', unsafe_allow_html=True)

page = st.sidebar.radio('Navigation', ['🔮 Prediction', '📊 Model Performance', 'ℹ️ About'])

if page == '🔮 Prediction':
    st.subheader('Booking Information')

    c1, c2, c3 = st.columns(3)
    with c1:
        hotel = st.selectbox('Hotel', ['Resort Hotel', 'City Hotel'])
        lead_time = st.number_input('Lead Time (days)', min_value=0, max_value=1000, value=60)
        arrival_year = st.number_input('Arrival Year', min_value=2015, max_value=2030, value=2017)
        arrival_month = st.selectbox('Arrival Month', ['January','February','March','April','May','June','July','August','September','October','November','December'])
        arrival_week = st.number_input('Arrival Week Number', min_value=1, max_value=53, value=27)
        arrival_day = st.number_input('Arrival Day', min_value=1, max_value=31, value=15)
        adr = st.number_input('ADR (Average Daily Rate)', min_value=0.0, max_value=10000.0, value=100.0, step=1.0)

    with c2:
        adults = st.number_input('Adults', min_value=0, max_value=50, value=2)
        children = st.number_input('Children', min_value=0.0, max_value=50.0, value=0.0, step=1.0)
        babies = st.number_input('Babies', min_value=0, max_value=20, value=0)
        meal = st.selectbox('Meal', ['BB', 'HB', 'FB', 'SC', 'Undefined'])
        country = st.text_input('Country Code', value='PRT')
        market_segment = st.selectbox('Market Segment', ['Direct','Corporate','Online TA','Offline TA/TO','Complementary','Groups','Undefined','Aviation'])
        distribution_channel = st.selectbox('Distribution Channel', ['Direct','Corporate','TA/TO','Undefined','GDS'])
        customer_type = st.selectbox('Customer Type', ['Transient','Contract','Transient-Party','Group'])

    with c3:
        is_repeated_guest = st.selectbox('Repeated Guest', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        previous_cancellations = st.number_input('Previous Cancellations', min_value=0, max_value=100, value=0)
        previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', min_value=0, max_value=100, value=0)
        reserved_room_type = st.text_input('Reserved Room Type', value='A')
        assigned_room_type = st.text_input('Assigned Room Type', value='A')
        booking_changes = st.number_input('Booking Changes', min_value=0, max_value=100, value=0)
        deposit_type = st.selectbox('Deposit Type', ['No Deposit','Non Refund','Refundable'])
        agent = st.number_input('Agent ID (0 = unknown)', min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        company = st.number_input('Company ID (0 = unknown)', min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        waiting = st.number_input('Days in Waiting List', min_value=0, max_value=400, value=0)
        parking = st.number_input('Required Car Parking Spaces', min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        special_requests = st.number_input('Special Requests', min_value=0.0, max_value=20.0, value=0.0, step=1.0)

    total_stay = st.number_input('Total Stay (nights)', min_value=0, max_value=100, value=3,
                                 help='The notebook derives this from weekend + weekday nights. In this UI we ask for the resulting total.')
    total_people = adults + children + babies
    total_bookings = previous_cancellations + previous_bookings_not_canceled
    total_previous_bookings = total_bookings

    st.divider()
    st.caption('The application automatically creates the engineered features used by the notebook: total_stay, total_people, total_bookings and total_previous_bookings.')

    if st.button('🔮 Predict Cancellation', type='primary', use_container_width=True):
        row = pd.DataFrame([{
            'hotel': hotel,
            'lead_time': lead_time,
            'arrival_date_year': arrival_year,
            'arrival_date_month': arrival_month,
            'arrival_date_week_number': arrival_week,
            'arrival_date_day_of_month': arrival_day,
            'adults': adults,
            'children': children,
            'babies': babies,
            'meal': meal,
            'country': country if country.strip() else 'Unknown',
            'market_segment': market_segment,
            'distribution_channel': distribution_channel,
            'is_repeated_guest': is_repeated_guest,
            'previous_cancellations': previous_cancellations,
            'previous_bookings_not_canceled': previous_bookings_not_canceled,
            'reserved_room_type': reserved_room_type.strip() or 'A',
            'assigned_room_type': assigned_room_type.strip() or 'A',
            'booking_changes': booking_changes,
            'deposit_type': deposit_type,
            'agent': agent,
            'company': company,
            'days_in_waiting_list': waiting,
            'customer_type': customer_type,
            'adr': adr,
            'required_car_parking_spaces': parking,
            'total_of_special_requests': special_requests,
            'total_stay': total_stay,
            'total_people': total_people,
            'total_bookings': total_bookings,
            'total_previous_bookings': total_previous_bookings,
        }])

        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1]) if hasattr(model, 'predict_proba') else None

        st.subheader('Prediction Result')
        if prediction == 1:
            st.error('🔴 Booking is likely to be CANCELED')
        else:
            st.success('🟢 Booking is likely to be NOT CANCELED')

        if probability is not None:
            st.metric('Cancellation Probability', f'{probability:.1%}')
            st.progress(probability)

        with st.expander('See model input'):
            st.dataframe(row, use_container_width=True)

elif page == '📊 Model Performance':
    st.subheader('XGBoost Model Performance')
    st.write('The deployed model is the XGBoost classifier used as the best model in the notebook comparison.')

    a, b, c, d = st.columns(4)
    a.metric('Accuracy', f"{metrics['accuracy']:.2%}")
    b.metric('Precision', f"{metrics['precision']:.2%}")
    c.metric('Recall', f"{metrics['recall']:.2%}")
    d.metric('F1 Score', f"{metrics['f1']:.2%}")

    st.info('The notebook comparison ranks XGBoost first by F1 Score: 0.8715 on the notebook\'s saved run. The local deployment model is retrained from the uploaded raw CSV using the same core pipeline, so its metrics can differ because the uploaded CSV contains a different row count than the notebook\'s saved run.')

elif page == 'ℹ️ About':
    st.subheader('About the Project')
    st.markdown('''
**Goal:** predict whether a hotel booking will be canceled.

**Target:** `is_canceled`

**Model:** XGBoost Classifier

**Preprocessing:**
- Numerical features → `StandardScaler`
- Categorical features → `OneHotEncoder(handle_unknown="ignore")`
- Both are combined with `ColumnTransformer` inside a `Pipeline`.

**Important:** `reservation_status` and `reservation_status_date` are excluded from prediction because they describe the final reservation outcome and would cause target leakage.
''')
