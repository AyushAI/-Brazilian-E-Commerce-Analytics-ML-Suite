import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    print("Loading data...")
    orders = pd.read_csv('olist_orders_dataset.csv')
    items = pd.read_csv('olist_order_items_dataset.csv')

    print("Merging datasets...")
    df = pd.merge(orders, items, on='order_id')

    print("Cleaning and standardizing...")
    # Filter for delivered orders (optional, but usually revenue is recognized on delivery or approval)
    df = df[df['order_status'] == 'delivered']
    
    # Convert timestamp to datetime
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Drop rows with missing values in key columns
    df.dropna(subset=['order_purchase_timestamp', 'price'], inplace=True)

    # Set as index for resampling
    df.set_index('order_purchase_timestamp', inplace=True)

    # Resample to monthly total revenue
    print("Aggregating to monthly revenue...")
    monthly_revenue = df['price'].resample('ME').sum()
    
    # Handle any potential remaining missing months if the data has gaps (fill with 0)
    monthly_revenue = monthly_revenue.fillna(0)
    
    return monthly_revenue

def check_stationarity(timeseries):
    print("\n--- Dickey-Fuller Test ---")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    if dftest[1] <= 0.05:
        print("Conclusion: Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary.")
        return True, timeseries
    else:
        print("Conclusion: Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.")
        print("Applying first-order differencing...")
        ts_diff = timeseries.diff().dropna()
        return False, ts_diff

def evaluate_models(train, test):
    print(f"\nTraining on {len(train)} months, Testing on {len(test)} months...")
    results = {}

    # --- ARIMA ---
    print("Training ARIMA...")
    try:
        # Order (p,d,q) chosen generically; in practice, use ACF/PACF or auto_arima
        arima_model = ARIMA(train, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=len(test))
        results['ARIMA'] = {
            'mae': mean_absolute_error(test, arima_pred),
            'rmse': np.sqrt(mean_squared_error(test, arima_pred)),
            'predictions': arima_pred
        }
    except Exception as e:
         print(f"ARIMA failed: {e}")

    # --- SARIMA ---
    print("Training SARIMA...")
    try:
        # Generic order and seasonal_order (assuming annual seasonality -> s=12)
        sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=len(test))
        results['SARIMA'] = {
            'mae': mean_absolute_error(test, sarima_pred),
            'rmse': np.sqrt(mean_squared_error(test, sarima_pred)),
            'predictions': sarima_pred
        }
    except Exception as e:
         print(f"SARIMA failed: {e}")

    # --- Prophet ---
    print("Training Prophet...")
    try:
       # Prophet requires specific column names: 'ds' (datetime) and 'y' (value)
       prophet_train = pd.DataFrame({'ds': train.index, 'y': train.values})
       prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
       prophet_model.fit(prophet_train)
       
       # Create future dataframe for testing period
       future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
       prophet_forecast = prophet_model.predict(future)
       
       # Extract only the test period predictions
       prophet_pred = prophet_forecast['yhat'].iloc[-len(test):].values
       
       results['Prophet'] = {
            'mae': mean_absolute_error(test, prophet_pred),
            'rmse': np.sqrt(mean_squared_error(test, prophet_pred)),
            'predictions': pd.Series(prophet_pred, index=test.index)
        }
    except Exception as e:
         print(f"Prophet failed: {e}")
         
    return results

def forecast_future(data, months=12):
    print(f"\nForecasting next {months} months using SARIMA (assuming it handles seasonality well)...")
    # Using SARIMA as default best for forecasting if it has seasonality
    # In a fully robust scenario, auto-select based on evaluate_models results
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=months)
    return forecast

def plot_results(data, test, results, future_forecast):
    plt.figure(figsize=(14, 8))
    
    # Plot historical
    plt.plot(data.index, data.values, label='Historical Revenue', color='black', linewidth=2)
    
    # Plot test predictions
    colors = ['red', 'blue', 'green']
    i = 0
    for name, metrics in results.items():
        if 'predictions' in metrics:
           plt.plot(test.index, metrics['predictions'], label=f'{name} (Test)', color=colors[i], linestyle='--')
           i += 1
           
    # Plot future forecast
    plt.plot(future_forecast.index, future_forecast.values, label='SARIMA Future Forecast', color='purple', linewidth=2)

    plt.title('E-commerce Monthly Revenue Forecast')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('revenue_forecast.png')
    print("\nSaved plot to revenue_forecast.png")

if __name__ == "__main__":
    revenue = load_and_prepare_data()
    print(f"\nTotal months of data: {len(revenue)}")
    print(revenue.head())
    
    if len(revenue) < 24: # Need enough data for seasonal splitting
       print("\nWarning: Not enough data for robust seasonal evaluation (ideally > 24 months). Proceeding anyway, but SARIMA might fail or overfit.")

    # 1. Stationarity Check
    is_stationary, diff_revenue = check_stationarity(revenue)
    
    # 2. Train/Test Split (last 6 months for testing is common for short datasets)
    test_size = 6
    if len(revenue) > test_size:
        train = revenue.iloc[:-test_size]
        test = revenue.iloc[-test_size:]
        
        # 3. Evaluate Models
        print("\n--- Evaluating Models ---")
        evaluation_results = evaluate_models(train, test)
        
        print("\n--- Model Performance ---")
        for name, metrics in evaluation_results.items():
             print(f"{name}: MAE={metrics.get('mae', 'N/A'):.2f}, RMSE={metrics.get('rmse', 'N/A'):.2f}")
             
        # 4. Forecast Future
        next_12_months = forecast_future(revenue, months=12)
        
        # 5. Plot
        plot_results(revenue, test, evaluation_results, next_12_months)
        
    else:
        print("\nError: Dataset is too short to split into train and test.")
