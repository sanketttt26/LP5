#include <iostream>
#include <omp.h>
#include <vector>
#include <climits>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()

using namespace std;

// Serial version of Min, Max, Sum, and Average
void serialOperations(const vector<int> &arr, int &min_val, int &max_val, int &sum_val, double &average_val)
{
    min_val = INT_MAX;
    max_val = INT_MIN;
    sum_val = 0;

    for (int i = 0; i < arr.size(); ++i)
    {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum_val += arr[i];
    }
    average_val = static_cast<double>(sum_val) / arr.size();
}

// Parallel version of Min, Max, Sum, and Average
void parallelOperations(const vector<int> &arr, int &min_val, int &max_val, int &sum_val, double &average_val)
{
    min_val = INT_MAX;
    max_val = INT_MIN;
    sum_val = 0;

#pragma omp parallel for reduction(min : min_val) reduction(max : max_val) reduction(+ : sum_val)
    for (int i = 0; i < arr.size(); ++i)
    {
        min_val = min(min_val, arr[i]);
        max_val = max(max_val, arr[i]);
        sum_val += arr[i];
    }

    average_val = static_cast<double>(sum_val) / arr.size();
}

int main()
{
    int n;
    cout << "Enter size of array: ";
    cin >> n;

    vector<int> arr(n);
    srand(time(0)); // Seed the random number generator

    // Generate random array elements
    for (int i = 0; i < n; ++i)
    {
        arr[i] = rand() % 1000; // Random values from 0 to 999
    }

    // Print generated array
    cout << "\nGenerated Array:\n";
    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Result variables
    int serial_min, serial_max, serial_sum;
    double serial_average;

    int parallel_min, parallel_max, parallel_sum;
    double parallel_average;

    double start, end;

    // Serial Operations
    start = omp_get_wtime();
    serialOperations(arr, serial_min, serial_max, serial_sum, serial_average);
    end = omp_get_wtime();
    cout << "\nSerial Version Time: " << end - start << " seconds\n";
    cout << "Serial Min: " << serial_min << ", Max: " << serial_max
         << ", Sum: " << serial_sum << ", Average: " << serial_average << endl;

    // Parallel Operations
    start = omp_get_wtime();
    parallelOperations(arr, parallel_min, parallel_max, parallel_sum, parallel_average);
    end = omp_get_wtime();
    cout << "\nParallel Version Time: " << end - start << " seconds\n";
    cout << "Parallel Min: " << parallel_min << ", Max: " << parallel_max
         << ", Sum: " << parallel_sum << ", Average: " << parallel_average << endl;

    return 0;
}
