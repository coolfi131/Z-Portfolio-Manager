
# Z-Portfolio-Manager

## How to build

Use python version 3.11 preferably

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

```bash
{
    "ACCOUNT_ID": "", 
    "PRIVATE_KEY": "",
    "ZCASH_NODE_URL": "",
    "ZCASH_USER": "",
    "ZCASH_PASS": "",
    "ZCASH_ACCOUNT_FILE": "",
    "ZCASH_ADDRESS": ""   ----> unified address only
}
```

## Installation

Install my-project with npm

```bash
  pip install -r requirements.txt
```
    
## Run 

```bash
  python3 main.py
```

## Setup Config

Sample File
```bash
  {
    "target_allocation": {
        "NEAR": 0.20,
        "ETH": 0.20,
        "BTC": 0.00,
        "USDC": 0.30,
        "SOL": 0.0,
        "DOGE": 0.00,
        "XRP": 0.00,
        "ZEC": 0.30
    },
    "rebalance_threshold": 0.07,
    "stop_loss": {
        "NEAR": 0.15,
        "ETH": 0.2,
        "BTC": 0.25,
        "SOL": 0.3,
        "DOGE": 0.5,
        "XRP": 0.5,
        "ZEC": 0.5
    },
    "take_profit": {
        "NEAR": 0.25,
        "ETH": 0.3,
        "BTC": 0.35,
        "SOL": 0.4,
        "DOGE": 0.6,
        "XRP": 0.6,
        "ZEC": 0.6
    }
}

```
