import pandas as pd
import matplotlib.pyplot as plt

## opening and loading some files
data = pd.read_parquet('2020-06-01/2020-06-01-TeamB-2f23d7d5-2326-49ce-b9c8-5a6303f785c5.parquet')
print(data.head())
data2 = pd.read_parquet('2020-06-01/2020-06-01-TeamB-8d723104-f773-83c1-3458-a748e9bb17bc.parquet')
print(data2.head())
## print(list(data))

## EDA - plotting against time
# Sample data columns provided
columns = ['speed', 'heart_rate', 'hacc', 'hdop', 'signal_quality', 
           'num_satellites', 'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Define a function to plot each column against time
def plot_columns_against_time(df):
    # Drop columns that can't be plotted (like player_name) and extract the rest
    for col in df.columns:
        if col != 'time' and col != 'player_name':
            plt.figure(figsize=(10, 6))
            plt.plot(df['time'], df[col], label=col)
            plt.title(f"{col} vs Time")
            plt.xlabel("Time")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.show()

plot_columns_against_time(data)