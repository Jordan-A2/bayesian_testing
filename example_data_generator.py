import pandas as pd
import numpy as np
from datetime import timedelta, date

# Generate date range
date_range = pd.date_range(start='2001-01-01', end='2001-04-01')
exp_groups = ['A', 'B', 'C']

# Initialize list for data rows
data = []

# Iterate over each experiment group and each date
for group in exp_groups:
    for send_date in date_range:
        # Random number of sends between 200,000 and 500,000
        sends = np.random.randint(200000, 500001)
        
        # Impressions: 95-99% of sends
        impressions = int(sends * np.random.uniform(0.95, 0.99))
        
        # Clicks: 2-8% of impressions
        clicks = int(impressions * np.random.uniform(0.02, 0.08))
        
        # Conversions: 75-125% of clicks
        conversions = int(clicks * np.random.uniform(0.75, 1.25))
        
        # Revenue: average revenue per conversion between 200 and 400 EUR
        attr_revenue_eur = float(conversions * np.random.uniform(200, 400))
        
        # Append the row to data
        data.append([group, send_date, impressions, clicks, conversions, attr_revenue_eur])

# Create DataFrame
columns = ['exp_group', 'sendDate', 'impressions', 'clicks', 'conversions', 'attr_revenue_eur']
df = pd.DataFrame(data, columns=columns)

# Sort by experiment group and date
df.sort_values(by=['exp_group', 'sendDate'], inplace=True)

# Save to CSV without 'sends' column
df.to_csv('example_data/example_data.csv', index=False)
