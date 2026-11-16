# Banking Transaction Analysis | NumPy

**A Python-based banking transaction analysis project using NumPy and synthetic financial data.**

## Project Overview

This project explores transaction amounts and customer-level daily transaction volumes through descriptive statistics, segmentation, outlier detection, and data validation. It demonstrates foundational and intermediate NumPy skills in a banking-related scenario.

## Technologies Used

- Python
- NumPy
- PyCharm
- GitHub

## Datasets

Both datasets are generated programmatically using NumPy with `np.random.seed(42)` for reproducibility.

| Dataset | Description |
|---|---|
| Part 1 | 1,000 synthetic individual transaction amounts |
| Part 2 | A matrix of 100 customers × 30 days; each cell represents one customer's daily transaction volume |
| Currency | Uzbekistani som (UZS) |

> **Note:** These are simulated data, not real bank or customer records. Part 1 and Part 2 are separate synthetic datasets.

## Part 1 — Bank Transaction Analysis

### Analytical Tasks

1. Inspect the number of transactions, data type, and first ten values.
2. Calculate total transaction volume, mean, minimum, and maximum.
3. Identify transactions greater than 4,000,000 UZS and calculate their count, total volume, share, and mean.
4. Segment transactions into low (below 1M), medium (1M–3M inclusive), and high (above 3M) amount bands.
5. Calculate median, standard deviation, and quartiles.
6. Identify statistical outliers using the interquartile range (IQR) rule.
7. Validate that the transaction categories account for all 1,000 records.

## Part 2 — Customer Transaction Matrix Analysis

### Analytical Tasks

1. Generate and inspect a 100 × 30 customer-by-day matrix.
2. Calculate each customer's 30-day total with `axis=1`.
3. Find the customer with the largest total transaction volume.
4. Calculate each day's total across customers with `axis=0`.
5. Find the highest-volume day and the average daily volume.
6. Count and compare days above and below the average.
7. Calculate the overall 30-day volume.
8. Verify that totals calculated by customer, by day, and from the full matrix agree.

## Key Analytical Results

Results below correspond to the fixed random seed and execution order in `main.py`.

| Metric | Result |
|---|---:|
| Individual transactions analyzed (Part 1) | 1,000 |
| Transactions above 4,000,000 UZS | 202 |
| IQR statistical outliers | 0 |
| Customers analyzed (Part 2) | 100 |
| Days analyzed (Part 2) | 30 |
| Highest-volume customer | Customer 17 |
| Highest-volume day | Day 21 |
| Total 30-day transaction volume (Part 2) | 1,536,955,275 UZS |

## Data Validation

- Confirm that transaction-segment counts sum to the original number of transactions.
- Compare customer-based totals, day-based totals, and the direct sum of the original matrix.
- Report `PASSED` or `FAILED` for these consistency checks.

These checks confirm agreement between the implemented aggregations; they do not independently establish the quality of source data.

## Skills Demonstrated

- NumPy array creation, indexing, and slicing
- Boolean masking and filtering
- Aggregation with `sum`, `mean`, `min`, `max`, and `argmax`
- Statistical analysis with median, standard deviation, and percentiles
- IQR-based outlier detection
- Two-dimensional arrays and `axis=0` / `axis=1`
- Basic data validation and formatted console reporting

## How to Run

1. Install Python 3 and NumPy:

   ```bash
   pip install numpy
   ```

2. Download or clone this repository.
3. Run the analysis from the repository folder:

   ```bash
   python main.py
   ```

## Limitations

This is an educational analysis of randomly generated data. A high-value transaction or an IQR outlier is **not** evidence of fraud. Transaction volume alone does not measure profitability. The 30-day simulation is not tied to a specific calendar month.

## Author

**Avaz — Data Analyst**

Python · NumPy · Pandas · SQL · Data Analysis

GitHub: https://github.com/avaz-analytics
