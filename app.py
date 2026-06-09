"""
Streamlit app — Bike Demand Forecaster.
========================================

CONTRACT — build a single-page Streamlit app with TWO parts:

1. Analytics dashboard: load the cleaned data and show >= 2 EDA visualisations
    (e.g. average demand by hour, demand vs. temperature, demand by weather situation).
2. Prediction form: collect weather + time inputs from the user, call
    ``src.predict.predict`` with the loaded champion, and display the predicted demand.

This is the file Hugging Face Spaces runs (see README front-matter: ``app_file: app.py``).
Keep it runnable with ``streamlit run app.py``.
"""
import streamlit as st
import plotly.express as px

from src.data_prep import add_features, load_and_clean
from src.predict import load_model, predict

DATA_PATH = "data/hour.csv"


@st.cache_data
def load_data():
    return add_features(load_and_clean(DATA_PATH))


@st.cache_resource
def load_champion():
    return load_model()


def main() -> None:
    st.set_page_config(page_title="Bike Demand Forecaster", page_icon="🚲", layout="wide")
    st.title("🚲 Bike Demand Forecaster")

    with st.spinner("Loading data..."):
        df = load_data()
    model = load_champion()

    tab1, tab2 = st.tabs(["📊 Analytics", "🔮 Predict"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            hourly = df.groupby("hr")["cnt"].mean().reset_index()
            fig1 = px.line(hourly, x="hr", y="cnt", markers=True,
                            title="Average Demand by Hour")
            fig1.update_layout(xaxis_title="Hour of day", yaxis_title="Avg rentals")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.scatter(df, x="temp", y="cnt", color="season",
                                title="Demand vs Temperature (by season)",
                                labels={"temp": "Temperature (norm.)", "cnt": "Rentals"})
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="weathersit", y="cnt",
                        title="Demand Distribution by Weather Situation",
                        labels={"weathersit": "Weather", "cnt": "Rentals"})
        fig3.update_xaxes(ticktext=["Clear", "Mist", "Light rain/snow", "Heavy rain/snow"],
                        tickvals=[1, 2, 3, 4])
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Enter weather and time details")

        with st.form("predict_form"):
            cols = st.columns(3)

            with cols[0]:
                hr = st.slider("Hour", 0, 23, 12)
                mnth = st.selectbox("Month", range(1, 13), format_func=lambda x: [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ][x - 1])
                season = st.selectbox("Season", [1, 2, 3, 4],
                                        format_func=lambda x: ["Winter", "Spring", "Summer", "Fall"][x - 1])
                yr = st.selectbox("Year", [0, 1], format_func=lambda x: "2011" if x == 0 else "2012")

            with cols[1]:
                weekday = st.selectbox("Day of week", range(0, 7),
                                        format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])
                holiday = st.checkbox("Holiday")
                workingday = st.checkbox("Working day")
                weathersit = st.selectbox("Weather", [1, 2, 3, 4],
                                        format_func=lambda x: [
                                            "Clear", "Mist", "Light rain/snow", "Heavy rain/snow"
                                        ][x - 1])

            with cols[2]:
                temp = st.slider("Temperature (norm.)", 0.0, 1.0, 0.5)
                atemp = st.slider("Feels-like temp (norm.)", 0.0, 1.0, 0.5)
                hum = st.slider("Humidity (norm.)", 0.0, 1.0, 0.5)
                windspeed = st.slider("Wind speed (norm.)", 0.0, 1.0, 0.2)

            submitted = st.form_submit_button("Predict demand", type="primary")

        if submitted:
            inputs = {
                "season": season, "yr": yr, "mnth": mnth, "hr": hr,
                "holiday": int(holiday), "weekday": weekday,
                "workingday": int(workingday), "weathersit": weathersit,
                "temp": temp, "atemp": atemp, "hum": hum, "windspeed": windspeed,
            }
            prediction = predict(model, inputs)
            st.metric("Predicted demand", f"{prediction:,.0f} rentals",
                    delta=None)

            st.caption("The model predicts total hourly rentals "
                    "(casual + registered) for the given conditions.")


if __name__ == "__main__":
    main()
