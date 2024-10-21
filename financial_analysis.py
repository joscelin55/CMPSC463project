import pandas as pd
import matplotlib.pyplot as plt
import math

# Load data from a CSV file
def load_data(your_data.cvs):
    return pd.read_csv(your_data)

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

# Kadane's Algorithm for Maximum Subarray
def max_subarray(arr):
    max_so_far = arr[0]
    max_ending_here = arr[0]
    for i in range(1, len(arr)):
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

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

# Plot trends
def plot_trends(data):
    plt.plot(data['date'], data['price'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Trends')
    plt.show()

# Main function
def main():
    data = load_data('your_data.csv')
    sorted_data = merge_sort(data['timestamp'].tolist())
    max_gain = max_subarray(data['price_change'].tolist())
    anomalies = closest_pair(data[['timestamp', 'price']].values.tolist())
    plot_trends(data)

if __name__ == "__main__":
    main()

