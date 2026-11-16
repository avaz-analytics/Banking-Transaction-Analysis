"""Banking Transaction Analysis — NumPy Portfolio Project.

Analyze synthetic transaction amounts and a customer-by-day transaction matrix.
All monetary values are expressed in Uzbekistani som (UZS).
"""

import numpy as np


# ======================================================================
# PART 1: BANK TRANSACTION ANALYSIS
# ======================================================================
# Dataset: 1,000 synthetic transaction amounts
# Objective: Explore transaction volumes, distributions, and outliers.
# ======================================================================

np.random.seed(42)
transactions = np.random.randint(10_000, 5_000_000, size=1_000)


# ----------------------------------------------------------------------
# 1. INITIAL DATA INSPECTION
# ----------------------------------------------------------------------
# Objectives:
# - Count transactions and inspect the dataset type.
# - Preview the first ten transaction amounts.

print('\n' + '=' * 62)
print('PART 1 | BANK TRANSACTION ANALYSIS')
print('=' * 62)
print('\n--- INITIAL DATA INSPECTION ---')
print(f'Total Transactions: {len(transactions):,}')
print(f'Data Type: {transactions.dtype}')
print(f'First 10 Transactions (UZS): {transactions[:10]}')


# ----------------------------------------------------------------------
# 2. FINANCIAL ANALYSIS
# ----------------------------------------------------------------------
# Objectives:
# - Calculate total transaction volume and average amount.
# - Find the minimum and maximum transaction amounts.

total = np.sum(transactions)
mean = np.mean(transactions)
max_tran = np.max(transactions)
min_tran = np.min(transactions)

print('\n--- FINANCIAL ANALYSIS ---')
print(f'Total Transaction Volume: {total:,.2f} UZS')
print(f'Average Transaction Amount: {mean:,.2f} UZS')
print(f'Largest Transaction: {max_tran:,.2f} UZS')
print(f'Smallest Transaction: {min_tran:,.2f} UZS')


# ----------------------------------------------------------------------
# 3. HIGH-VALUE TRANSACTION ANALYSIS
# ----------------------------------------------------------------------
# Objectives:
# - Identify transactions above 4,000,000 UZS.
# - Measure their count, volume, shares, and average amount.
# High-value transactions are defined as amounts above 4,000,000 UZS.
# This threshold is for segmentation, not fraud detection.

high_transactions = transactions[transactions > 4_000_000]
high_total = np.sum(high_transactions)
high_percent = len(high_transactions) / len(transactions) * 100
high_volume_percent = high_total / total * 100 if total else 0.0
high_average = np.mean(high_transactions) if len(high_transactions) else 0.0

print('\n--- HIGH-VALUE TRANSACTION ANALYSIS ---')
print(f'High-Value Transaction Count: {len(high_transactions):,}')
print(f'High-Value Transaction Volume: {high_total:,.2f} UZS')
print(f'Share of Transaction Count: {high_percent:.2f}%')
print(f'Share of Total Volume: {high_volume_percent:.2f}%')
print(f'Average High-Value Transaction: {high_average:,.2f} UZS')


# ----------------------------------------------------------------------
# 4. TRANSACTION DISTRIBUTION
# ----------------------------------------------------------------------
# Objectives:
# - Segment amounts into low, medium, and high bands.
# - Calculate category shares and verify that all records are counted.
# Low: < 1M | Medium: 1M–3M (inclusive) | High: > 3M UZS

below_1mln = transactions[transactions < 1_000_000]
between_1mln_3mln = transactions[
    (transactions >= 1_000_000) & (transactions <= 3_000_000)
]
above_3mln = transactions[transactions > 3_000_000]

percent_low = len(below_1mln) / len(transactions) * 100
percent_medium = len(between_1mln_3mln) / len(transactions) * 100
percent_high = len(above_3mln) / len(transactions) * 100

verified_count = (
    len(below_1mln) + len(between_1mln_3mln) + len(above_3mln)
)

print('\n--- TRANSACTION DISTRIBUTION ---')
print(f'Low (< 1M UZS): {len(below_1mln):,} ({percent_low:.2f}%)')
print(f'Medium (1M–3M UZS): {len(between_1mln_3mln):,} ({percent_medium:.2f}%)')
print(f'High (> 3M UZS): {len(above_3mln):,} ({percent_high:.2f}%)')
print(f'Transactions Accounted For: {verified_count:,}')
print(f'Original Transaction Count: {len(transactions):,}')
print(f'Count Validation: {"PASSED" if verified_count == len(transactions) else "FAILED"}')


# ----------------------------------------------------------------------
# 5. STATISTICAL ANALYSIS
# ----------------------------------------------------------------------
# Objectives:
# - Measure median, standard deviation, and quartiles.
# - Compare the mean and median transaction amounts.

median = np.median(transactions)
std_dev = np.std(transactions)
q1 = np.percentile(transactions, 25)
q3 = np.percentile(transactions, 75)

print('\n--- STATISTICAL ANALYSIS ---')
print(f'Mean Transaction Amount: {mean:,.2f} UZS')
print(f'Median Transaction Amount: {median:,.2f} UZS')
print(f'Standard Deviation: {std_dev:,.2f} UZS')
print(f'25th Percentile (Q1): {q1:,.2f} UZS')
print(f'75th Percentile (Q3): {q3:,.2f} UZS')

if mean > median:
    print('Mean vs Median: Mean is greater than median')
elif mean < median:
    print('Mean vs Median: Median is greater than mean')
else:
    print('Mean vs Median: Mean and median are equal')


# ----------------------------------------------------------------------
# 6. OUTLIER DETECTION (IQR METHOD)
# ----------------------------------------------------------------------
# Objectives:
# - Calculate IQR-based lower and upper limits.
# - Count transactions outside those limits and report their share.

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = transactions[
    (transactions < lower_bound) | (transactions > upper_bound)
]
outliers_count = len(outliers)
outliers_percent = outliers_count / len(transactions) * 100

print('\n--- OUTLIER DETECTION ---')
print(f'Interquartile Range (IQR): {iqr:,.2f} UZS')
print(f'Lower Bound: {lower_bound:,.2f} UZS')
print(f'Upper Bound: {upper_bound:,.2f} UZS')
print(f'Detected Outliers: {outliers_count:,}')
print(f'Outlier Percentage: {outliers_percent:.2f}%')
print(
    'Outlier Analysis: No statistical outliers detected'
    if outliers_count == 0
    else 'Outlier Analysis: Statistical outliers detected'
)


# ======================================================================
# PART 2: CUSTOMER TRANSACTION MATRIX ANALYSIS
# ======================================================================
# Dataset: 100 customers x 30 days
# Rows represent customers; columns represent days.
# Each cell represents a customer's total transaction volume for a day.
# ======================================================================

customer_transactions = np.random.randint(
    10_000, 1_000_001, size=(100, 30)
)


# ----------------------------------------------------------------------
# 1. DATA OVERVIEW
# ----------------------------------------------------------------------
# Objectives:
# - Generate a 100-customer by 30-day matrix.
# - Inspect its dimensions and number of daily observations.

print('\n' + '=' * 62)
print('PART 2 | CUSTOMER TRANSACTION MATRIX ANALYSIS')
print('=' * 62)
print('\n--- DATA OVERVIEW ---')
print(f'Dataset Shape: {customer_transactions.shape}')
print(f'Total Customers: {customer_transactions.shape[0]}')
print(f'Total Days: {customer_transactions.shape[1]}')
print(f'Total Daily Observations: {customer_transactions.size:,}')


# ----------------------------------------------------------------------
# 2. CUSTOMER TRANSACTION ANALYSIS
# ----------------------------------------------------------------------
# Objectives:
# - Sum daily volumes for each customer using axis=1.
# - Identify the customer with the highest 30-day volume.

customer_totals = np.sum(customer_transactions, axis=1)
top_customer_index = np.argmax(customer_totals)
max_customer_total = np.max(customer_totals)

print('\n--- CUSTOMER TRANSACTION ANALYSIS ---')
print(f'Total Customers Analyzed: {len(customer_totals):,}')
print(f'Top Customer ID: {top_customer_index + 1}')
print(f'Top Customer Transaction Volume: {max_customer_total:,} UZS')


# ----------------------------------------------------------------------
# 3. DAILY TRANSACTION ANALYSIS
# ----------------------------------------------------------------------
# Objectives:
# - Sum customer volumes for each day using axis=0.
# - Find the highest-volume day and average daily volume.

daily_totals = np.sum(customer_transactions, axis=0)
max_day = np.argmax(daily_totals)
max_daily_volume = np.max(daily_totals)
avg_daily_volume = np.mean(daily_totals)

print('\n--- DAILY TRANSACTION ANALYSIS ---')
print(f'Total Days Analyzed: {len(daily_totals)}')
print(f'Highest Volume Day: Day {max_day + 1}')
print(f'Max Daily Volume: {max_daily_volume:,} UZS')
print(f'Average Daily Volume: {avg_daily_volume:,.2f} UZS')


# ----------------------------------------------------------------------
# 4. ABOVE- AND BELOW-AVERAGE DAILY VOLUME
# ----------------------------------------------------------------------
# Objectives:
# - Filter days above and below the average daily volume.
# - Count these days and calculate their percentages.

days_below_avg = daily_totals[daily_totals < avg_daily_volume]
days_above_avg = daily_totals[daily_totals > avg_daily_volume]

count_below_avg = len(days_below_avg)
count_above_avg = len(days_above_avg)
count_equal_avg = np.count_nonzero(daily_totals == avg_daily_volume)

percent_below_avg = count_below_avg / len(daily_totals) * 100
percent_above_avg = count_above_avg / len(daily_totals) * 100

print('\n--- DAILY VOLUME VS AVERAGE ---')
print(f'Days Above Average: {count_above_avg} ({percent_above_avg:.2f}%)')
print(f'Days Below Average: {count_below_avg} ({percent_below_avg:.2f}%)')
print(f'Days Equal to Average: {count_equal_avg}')


# ----------------------------------------------------------------------
# 5. MONTHLY TRANSACTION SUMMARY
# ----------------------------------------------------------------------
# Objectives:
# - Calculate the total 30-day transaction volume.
# - Summarize average and peak daily volumes.

monthly_volume = np.sum(daily_totals)

print('\n--- MONTHLY TRANSACTION SUMMARY ---')
print(f'Total Monthly Volume: {monthly_volume:,} UZS')
print(f'Average Daily Volume: {avg_daily_volume:,.2f} UZS')
print(f'Highest Daily Volume: {max_daily_volume:,} UZS')


# ----------------------------------------------------------------------
# 6. FINAL DATA VALIDATION
# ----------------------------------------------------------------------
# Objectives:
# - Calculate totals by customer, by day, and from the matrix.
# - Verify that all three totals agree.

total_by_customers = np.sum(customer_totals)
total_by_days = np.sum(daily_totals)
total_from_matrix = np.sum(customer_transactions)

print('\n--- FINAL DATA VALIDATION ---')
print(f'Total by Customers: {total_by_customers:,} UZS')
print(f'Total by Days: {total_by_days:,} UZS')
print(f'Total from Original Matrix: {total_from_matrix:,} UZS')

if total_by_customers == total_by_days == total_from_matrix:
    print('Validation Status: PASSED')
else:
    print('Validation Status: FAILED')

print('=' * 62)






