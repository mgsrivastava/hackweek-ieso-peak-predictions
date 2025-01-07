# Ontario Power Demand Prediction

This web application visualizes historical power demand data for Ontario and provides predictions for peak demand days in 2024 using machine learning models.

## Features

- Historical demand visualization (2021-2023)
- Weather data visualization for 2024
- Machine learning predictions for 2024 peak demand days
- Top 20 predicted peak demand days (excluding weekends)

## Data Sources

- Historical demand data from the IESO Power Data Directory
- Weather data from Environment Canada

## Technology Stack

- Frontend: HTML, JavaScript, Chart.js
- Backend: Python (data processing and machine learning)
- Machine Learning: Random Forest Regressor

## Model Performance

The Random Forest model achieved an RMSE of approximately 1626.70 MW on the validation set, demonstrating strong predictive performance for daily demand patterns.

## Usage

The application is live at: https://meghnasrivastava.github.io/peak_prediction/

This visualization tool allows you to:

- Compare historical demand patterns across 2021-2023
- View temperature forecasts for 2024
- See predicted peak demand days for 2024
- Analyze the top 20 predicted peak demand days (excluding weekends)

## Local Development

To run the application locally:

1. Clone the repository
2. Run a local web server:
   ```bash
   python -m http.server 8000
   ```
3. Open `http://localhost:8000` in your browser

## Data Updates

The weather data and predictions can be updated by running:

```bash
python predict_demand.py
```

After updating predictions, commit and push the changes to GitHub to update the live site:

```bash
git add predictions.json
git commit -m "Update predictions"
git push
```
