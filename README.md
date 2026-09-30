# DBSCAN + NetworkX Transaction Anomaly Detection

## Project Overview

This project demonstrates how **DBSCAN clustering** and **NetworkX graph
analysis** can be combined to investigate unusual transaction behavior in a
large-data setting.

The project uses a reproducible synthetic transaction dataset. DBSCAN groups
transactions according to behavioral features and labels observations outside
the identified dense regions as noise (`cluster = -1`). NetworkX then models
relationships among customers, transactions, and merchants so that the
relationships surrounding flagged observations can be inspected.

> Important: A DBSCAN noise label means that an observation does not fit the
> selected density pattern. It does not prove fraud or another real-world
> problem.

## Dataset

The dataset is intentionally synthetic so that the project can be reproduced
without exposing private financial information.

File:

`data/transactions.csv`

There are 300 transaction records with the following main variables:

- `transaction_id`: unique transaction identifier
- `customer_id`: customer associated with the transaction
- `merchant_id`: merchant associated with the transaction
- `device_id`: device identifier
- `amount`: transaction amount
- `transactions_24h`: number of transactions associated with the customer in
  the previous 24 hours
- `unique_merchants_7d`: number of merchants used during the previous 7 days
- `location_changes_7d`: number of location changes during the previous 7 days
- `device_changes_30d`: number of device changes during the previous 30 days
- `known_pattern`: synthetic ground-truth label used only to evaluate the
  demonstration; it is NOT supplied to DBSCAN

## Method

### 1. Feature selection

DBSCAN uses:

- transaction amount
- transactions in 24 hours
- unique merchants in 7 days
- location changes in 7 days
- device changes in 30 days

### 2. Standardization

The features have different numerical scales. `StandardScaler` is therefore
used before DBSCAN so that a variable with a larger numerical range does not
dominate the distance calculation.

### 3. DBSCAN

The demonstration uses:

- `eps = 1.15`
- `min_samples = 7`

DBSCAN identifies dense regions and assigns `-1` to observations classified as
noise.

### 4. NetworkX

NetworkX represents a transaction network using:

`Customer -> Transaction -> Merchant`

The graph is then used to inspect the customers and merchants directly
connected to DBSCAN-flagged transactions.

## Visualizations

The project produces two main figures:

1. `results/dbscan_anomalies.png`
   - Shows transaction amount against transaction frequency.
   - DBSCAN noise observations are marked separately.

2. `results/networkx_anomaly_subgraph.png`
   - Shows the network around transactions flagged by DBSCAN.
   - Customers, merchants, transactions, and anomaly transactions are
     represented as different node shapes.

## How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd dbscan-networkx-anomaly-project
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

```bash
python src/anomaly_detection.py
```

The generated CSV files and figures will appear in the `results/` directory.

## Academic Interpretation

DBSCAN is useful for anomaly-oriented clustering because it does not require
the number of clusters to be specified in advance and can identify noise
observations. The method was originally introduced by Ester et al. (1996).

NetworkX is used for the graph component because it provides data structures
and algorithms for network analysis. In this project, DBSCAN performs the
density-based clustering while NetworkX represents and analyzes relationships.
They therefore serve different but complementary purposes.

## Limitations

This is an educational demonstration rather than a fraud-detection system.
The dataset is synthetic, the DBSCAN parameters are fixed for demonstration,
and the synthetic anomaly labels are constructed rather than obtained from a
real financial institution.

Real-world applications would require domain-specific feature engineering,
parameter validation, handling of changing data distributions, and appropriate
evaluation against independently verified cases.

## Suggested Repository Structure

```text
dbscan-networkx-anomaly-project/
├── data/
│   └── transactions.csv
├── results/
│   ├── dbscan_anomalies.png
│   ├── networkx_anomaly_subgraph.png
│   ├── evaluation_summary.csv
│   ├── network_summary.csv
│   ├── transaction_network.edgelist
│   └── transactions_with_dbscan_labels.csv
├── src/
│   └── anomaly_detection.py
├── .gitignore
├── README.md
└── requirements.txt
```
