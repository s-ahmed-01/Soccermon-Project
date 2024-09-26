from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from math import radians, sin, cos, sqrt, atan2

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AthleteDailyActivity") \
    .master("local[*]") \
    .getOrCreate()

# Load parquet files
data_dir = "Parquets/2020/2020-06"
df = spark.read.parquet(f"{data_dir}/*.parquet")

# Define a UDF (User-Defined Function) to calculate distance using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Radius of Earth in meters
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Register the haversine function as a UDF
from pyspark.sql.types import DoubleType
spark.udf.register("haversine", haversine, DoubleType())

# Calculate total gyro change, total acceleration change, and distance between successive points
df = df.withColumn('accel_change', F.sqrt(F.pow(F.col('accl_x'), 2) + F.pow(F.col('accl_y'), 2) + F.pow(F.col('accl_z'), 2))) \
       .withColumn('gyro_change', F.sqrt(F.pow(F.col('gyro_x'), 2) + F.pow(F.col('gyro_y'), 2) + F.pow(F.col('gyro_z'), 2)))

# Shift the lat/lon columns by one row to calculate the distance
window = Window.partitionBy("player_name").orderBy("time")
df = df.withColumn("lat_shifted", F.lag(df["lat"]).over(window)) \
       .withColumn("lon_shifted", F.lag(df["lon"]).over(window))

# Calculate distance using Haversine formula
df = df.withColumn("distance", F.expr("haversine(lat, lon, lat_shifted, lon_shifted)"))

# Group by player and date (assuming 'time' column includes date and time)
df = df.withColumn('date', F.to_date(F.col('time')))

# Aggregate the metrics for each player on each day
df_daily = df.groupBy("player_name", "date") \
    .agg(
        F.sum("gyro_change").alias("total_gyro_change"),
        F.sum("accel_change").alias("total_accel_change"),
        F.sum("distance").alias("total_distance_covered"),
        F.avg("heart_rate").alias("avg_heart_rate")
    )

# Show the final daily activity summary for each player
df_daily.show()

# Stop the Spark session
spark.stop()
