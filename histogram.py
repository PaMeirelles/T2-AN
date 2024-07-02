import numpy as np
import csv

# Function to read data from CSV and compute percentiles
def read_csv_and_compute_percentiles(filename, percentiles_range):
    iterations = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            iterations.append(int(row['iterations']))  # Assuming 'iterations' is integer

    # Compute specified percentiles
    percentiles = [np.percentile(iterations, p) for p in percentiles_range]

    return percentiles

# Example filenames (replace with your actual filenames)
filename1 = 'f1_replace_oldest_1e-5_tol_1e-5_alpha_1e-2_range_1000.csv'
filename2 = 'f1_replace_oldest_1e-5_tol_1e-1_delta_1e-2_range_1000.csv'

# Specify the range of percentiles to compute
percentiles_range = range(0, 101, 10)  # Computes percentiles 10, 20, 30, ..., 100

# Read and compute specified percentiles for dataset 1
percentiles_data1 = read_csv_and_compute_percentiles(filename1, percentiles_range)

# Read and compute specified percentiles for dataset 2
percentiles_data2 = read_csv_and_compute_percentiles(filename2, percentiles_range)

# Displaying the percentiles in a table format
print(f"Percentiles for Dataset 1 ({filename1}):")
print("Percentile\tValue")
for i, percentile in zip(percentiles_range, percentiles_data1):
    print(f"{i}%\t\t{percentile}")

print("\n")

print(f"Percentiles for Dataset 2 ({filename2}):")
print("Percentile\tValue")
for i, percentile in zip(percentiles_range, percentiles_data2):
    print(f"{i}%\t\t{percentile}")
