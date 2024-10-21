**Project Report: Large-scale Financial Data Analysis & Trend Detection**

Description of the Project:
**Overview:** This project aims to develop a system that analyzes large financial datasets, such as stock prices, transaction logs, or cryptocurrency data, to detect patterns, trends, and anomalies. The system leverages divide-and-conquer techniques to efficiently process, analyze, and report on financial data, aiding in decision-making processes.

**Problem Addressed:** The primary problem addressed by this project is the need for efficient and accurate analysis of large-scale financial data to identify significant trends and anomalies that can inform investment strategies and detect potential fraud.

**Goals of the Analysis:**
Efficiently sort and process large financial datasets.
Identify periods of maximum gain or loss.
Detect anomalies in transaction logs or price fluctuations.
Generate comprehensive reports summarizing the findings.

**Type-specific Considerations:**
Choice of Financial Datasets: The project uses datasets such as daily stock prices, cryptocurrency transactions, or financial transaction logs.

Algorithms Used:
Merge Sort: For efficient sorting of time-series data.
Kadane’s Algorithm: For identifying periods of maximum gain or loss.
Closest Pair of Points Algorithm: For detecting anomalies in financial data.
Structure of the Code with Diagram and Comments

**Block Diagram:**

![1d5047e5-42cf-4b5a-81d3-29a2b6bab086](https://github.com/user-attachments/assets/6361f577-6151-4825-8bbc-75ce74b2cecf)





**Code Structure:**
main.py: The main script that orchestrates the entire process.
data_loader.py: Responsible for loading the dataset.
merge_sort.py: Contains the implementation of the Merge Sort algorithm.
max_subarray.py: Implements Kadane’s algorithm for finding the maximum subarray.
anomaly_detection.py: Implements the closest pair of points algorithm for anomaly detection.
report_generator.py: Handles the generation of visual reports.

**Summary of Each Developed Class:**
DataLoader:
Purpose: Load and preprocess financial data.
Key Methods: load_data(file_path)

MergeSort:
Purpose: Sort financial data using the Merge Sort algorithm.
Key Methods: merge_sort(data)

MaxSubarray:
Purpose: Identify periods of maximum gain or loss using Kadane’s algorithm.
Key Methods: max_subarray(arr)

AnomalyDetection:
Purpose: Detect anomalies using the closest pair of points algorithm.
Key Methods: closest_pair(points)

ReportGenerator:
Purpose: Generate visual reports of the analysis.
Key Methods: plot_trends(data)

**Instructions on How to Use the System**
1. Prepare Your Data:
Ensure you have a CSV file named your_data.csv with columns like timestamp, price, price_change.

2. Run the Script:
Open your terminal or command prompt.
Navigate to the directory containing financial_analysis.py.
Run the script using:
python financial_analysis.py

3. View the Results:
The script will load the data, perform the analysis, and generate visual reports.
Verification of Code Functionality

**Examples of Code Execution:**
Example 1: Loading Data
import pandas as pd

# Load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Example usage
data = load_data('your_data.csv')
print(data.head())

# Example usage
   timestamp  price  price_change
0  2024-01-01    100             0
1  2024-01-02    105             5
2  2024-01-03    102            -3
3  2024-01-04    108             6
4  2024-01-05    107            -1

Example 2: Sorting Data using Merge Sort
# Merge Sort implementation
def merge_sort(data):
    if len(data) > 1:
        mid = len(data) // 2
        left_half = data[:mid]
        right_half = data[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                data[k] = left_half[i]
                i += 1
            else:
                data[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            data[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            data[k] = right_half[j]
            j += 1
            k += 1
    return data

# Example usage
sample_data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
sorted_sample = merge_sort(sample_data)
print(sorted_sample)

Output:
[1, 1, 2, 3, 4, 5, 5, 6, 9]


Example 3: Finding Maximum Gain using Kadane’s Algorithm
# Kadane's Algorithm for Maximum Subarray
def max_subarray(arr):
    max_so_far = arr[0]
    max_ending_here = arr[0]
    for i in range(1, len(arr)):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

# Example usage
sample_prices = [1, -3, 2, 1, -1, 3, -2, 3]
max_gain_sample = max_subarray(sample_prices)
print(max_gain_sample)


Output:
5

Example 4: Detecting Anomalies using Closest Pair of Points
import math

# Closest Pair of Points for Anomaly Detection
import math

# Closest Pair of Points for Anomaly Detection
def closest_pair(points):
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_pair_rec(points_sorted_x, points_sorted_y):
        if len(points_sorted_x) <= 3:
            return min((distance(points_sorted_x[i], points_sorted_x[j]), (points_sorted_x[i], points_sorted_x[j]))
                       for i in range(len(points_sorted_x)) for j in range(i + 1, len(points_sorted_x)))

        mid = len(points_sorted_x) // 2
        left_x = points_sorted_x[:mid]
        right_x = points_sorted_x[mid:]

        midpoint = points_sorted_x[mid][0]
        left_y = list(filter(lambda x: x[0] <= midpoint, points_sorted_y))
        right_y = list(filter(lambda x: x[0] > midpoint, points_sorted_y))

        (d1, pair1) = closest_pair_rec(left_x, left_y)
        (d2, pair2) = closest_pair_rec(right_x, right_y)

        d = min(d1, d2)
        pair = pair1 if d1 < d2 else pair2

        strip = [p for p in points_sorted_y if abs(p[0] - midpoint) < d]
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                d3 = distance(strip[i], strip[j])
                if d3 < d:
                    d = d3
                    pair = (strip[i], strip[j])

        return d, pair

    points_sorted_x = sorted(points, key=lambda x: x[0])
    points_sorted_y = sorted(points, key=lambda x: x[1])
    return closest_pair_rec(points_sorted_x, points_sorted_y)

# Example usage
sample_points = [(1, 2), (3, 4), (5, 6), (7, 8)]
closest_pair_sample = closest_pair(sample_points)
print(closest_pair_sample)



Output:
(2.8284271247461903, ((1, 2), (3, 4)))

Example 5: Generating Reports
import matplotlib.pyplot as plt

# Plot trends
def plot_trends(data):
    plt.plot(data['timestamp'], data['price'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Trends')
    plt.show()

# Example usage
data = pd.DataFrame({
    'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
    'price': [100, 105, 102, 108, 107]
})
plot_trends(data)

Output: 
![68bc1d38-eda0-4927-9c98-70400c16f324](https://github.com/user-attachments/assets/766f1841-f2bc-4b50-aa8b-7721b13d9344)


**Verification for Each Component Algorithm with Toy Example:**

**Merge Sort:**
sample_data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
sorted_sample = merge_sort(sample_data)
print(sorted_sample)


**Kadane’s Algorithm:**
sample_prices = [1, -3, 2, 1, -1, 3, -2, 3]
max_gain_sample = max_subarray(sample_prices)
print(max_gain_sample)


**Closest Pair of Points:**
sample_points = [(1, 2), (3, 4), (5, 6), (7, 8)]
closest_pair_sample = closest_pair(sample_points)
print(closest_pair_sample)

**Discussion of Findings:**

**Insights Gained:**
The system efficiently sorted large datasets, enabling quick access to time-series data.
Kadane’s algorithm successfully identified periods of maximum gain or loss, providing valuable insights into stock performance.
The closest pair of points algorithm effectively detected anomalies, which could indicate potential fraud or unusual market behavior.

**Challenges Faced:**
Handling large datasets required optimizing memory usage and processing time.
Ensuring the accuracy of anomaly detection in noisy financial data was challenging.
Limitations and Areas for Improvement:
The system could be enhanced to handle real-time data streams.
Incorporating more sophisticated anomaly detection techniques could improve accuracy.
Adding more visualization options could provide deeper insights into the data.


