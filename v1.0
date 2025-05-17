import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import numpy as np
from utils import fetch_historical_weather_data

# Configure the page
st.set_page_config(
    page_title="Historical Temperature Visualizer",
    page_icon="🌡️",
    layout="wide"
)

# App title and description
st.title("Historical Temperature Visualizer")
st.markdown("""
    This application visualizes the maximum daily temperatures for a specific location and month 
    over a 30-year period. Enter a location and select a month to see how temperatures have 
    changed over time.
""")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Input for location
location_input_type = st.sidebar.radio(
    "Location Input Type",
    ["City Name", "Coordinates"]
)

if location_input_type == "City Name":
    city_name = st.sidebar.text_input("Enter City Name", "New York")
    if city_name:
        # Initialize geolocator
        geolocator = Nominatim(user_agent="temp_visualizer")
        try:
            # Get coordinates from city name
            location = geolocator.geocode(city_name)
            if location:
                latitude, longitude = location.latitude, location.longitude
                st.sidebar.success(f"Located: {location.address}")
                st.sidebar.write(f"Latitude: {latitude}, Longitude: {longitude}")
            else:
                st.sidebar.error("Location not found. Please try a different name.")
                latitude, longitude = None, None
        except (GeocoderTimedOut, GeocoderUnavailable):
            st.sidebar.error("Geocoding service timed out. Please try again later.")
            latitude, longitude = None, None
    else:
        latitude, longitude = None, None
else:  # Coordinates input
    col1, col2 = st.sidebar.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=40.7128, min_value=-90.0, max_value=90.0, step=0.0001)
    with col2:
        longitude = st.number_input("Longitude", value=-74.0060, min_value=-180.0, max_value=180.0, step=0.0001)

# Month selection
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
selected_month = st.sidebar.selectbox("Select Month", months)
month_number = months.index(selected_month) + 1

# Button to trigger visualization
if st.sidebar.button("Visualize Temperature Data"):
    if latitude is not None and longitude is not None:
        try:
            with st.spinner("Fetching and processing historical temperature data..."):
                # Current year for calculating the 30-year period
                current_year = datetime.datetime.now().year
                start_year = current_year - 30
                
                # Call the function to fetch historical weather data
                df = fetch_historical_weather_data(latitude, longitude, month_number, start_year)
                
                if df is not None and not df.empty:
                    # Display summary statistics
                    st.subheader(f"Summary Statistics for {selected_month}")
                    
                    # Calculate average max temp per year for the selected month
                    yearly_avg = df.groupby('year')['max_temp'].mean().reset_index()
                    
                    # Calculate the trend (linear regression)
                    X = yearly_avg['year'].values.reshape(-1, 1)
                    y = yearly_avg['max_temp'].values
                    
                    # Simple linear regression
                    slope = np.polyfit(yearly_avg['year'], yearly_avg['max_temp'], 1)[0]
                    
                    # Create two columns for stats and trend info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Overall Avg Max Temp", f"{df['max_temp'].mean():.1f}°C")
                        st.metric("Highest Recorded Temp", f"{df['max_temp'].max():.1f}°C")
                        st.metric("Lowest Recorded Temp", f"{df['max_temp'].min():.1f}°C")
                    
                    with col2:
                        trend_direction = "Rising" if slope > 0 else "Falling" if slope < 0 else "Stable"
                        trend_magnitude = abs(slope * 10)  # Magnitude of change per decade
                        
                        st.metric(
                            "Temperature Trend", 
                            f"{trend_direction} at {trend_magnitude:.2f}°C per decade"
                        )
                        st.metric("Total Years of Data", len(yearly_avg))
                    
                    # Create heatmap visualization
                    st.subheader(f"Daily Maximum Temperatures for {selected_month} (1991-{current_year})")
                    
                    # Pivot the data for the heatmap: days on y-axis, years on x-axis
                    heatmap_data = df.pivot(index='day', columns='year', values='max_temp')
                    
                    # Create a Plotly figure
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Year", y="Day of Month", color="Max Temp (°C)"),
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        color_continuous_scale='RdYlBu_r',  # Red-Yellow-Blue reversed (red is hot)
                        aspect="auto"
                    )
                    
                    fig.update_layout(
                        height=600,
                        xaxis=dict(tickangle=45),
                        title=f"Maximum Daily Temperatures for {selected_month} ({start_year}-{current_year})"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Line chart showing yearly averages
                    st.subheader(f"Yearly Average Maximum Temperature for {selected_month}")
                    
                    fig_line = px.line(
                        yearly_avg, 
                        x='year', 
                        y='max_temp',
                        markers=True,
                        labels={"max_temp": "Average Max Temp (°C)", "year": "Year"},
                        title=f"Trend of Average Maximum Temperature for {selected_month} ({start_year}-{current_year})"
                    )
                    
                    # Add a trend line
                    fig_line.add_traces(
                        px.scatter(
                            x=yearly_avg['year'],
                            y=yearly_avg['year'] * slope + np.polyfit(yearly_avg['year'], yearly_avg['max_temp'], 1)[1],
                            trendline="ols",
                            labels={"x": "Year", "y": "Trend"}
                        ).data[1]
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Display the raw data in an expandable section
                    with st.expander("View Raw Data"):
                        st.dataframe(df)
                        
                        # Allow downloading the data as CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name=f'temperature_data_{selected_month}_{start_year}_{current_year}.csv',
                            mime='text/csv',
                        )
                else:
                    st.error("No data available for the selected location and month. Please try a different selection.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid location before visualizing data.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Data source: Open-Meteo API")
st.sidebar.caption("© 2023 Historical Temperature Visualizer")
