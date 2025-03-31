import json
import requests
from decimal import Decimal, ROUND_HALF_DOWN
from rich.console import Console
from rich.markdown import Markdown
import os

from typing import List, Optional, Union
from py_near.constants import DEFAULT_ATTACHED_GAS
import zcash
from intents import NearAccount

default_mainnet_rpc = "https://rpc.mainnet.near.org"
user_account_id = os.environ.get("ACCOUNT_ID")
user_private_key = os.environ.get("PRIVATE_KEY")

NEAR_BUFFER = 25000000000000000000000

with open("env.json", "r") as file:
    env_vars = json.load(file)

with open("tokens.json", "r") as file:
    data = json.load(file)

def load_url(url):
    r = requests.get(url, timeout=2)
    r.raise_for_status()
    return r.json()

async def _wallet_balance(account_id, data_old):
    data = (load_url("https://api-mng-console.chaindefuser.com/api/tokens"))
    if data["items"] is None:
        data = data_old
    else:   
        data = data["items"]

    try:
        # Fetch tokens (excluding NEAR)
        token_response = requests.get(f"https://api.fastnear.com/v1/account/{account_id}/ft")
        token_response.raise_for_status()  # Raise an exception for bad status codes

        
        # Fetch NEAR balance
        near_response = requests.get(f"https://api.nearblocks.io/v1/account/{account_id}")
        near_response.raise_for_status()  # Raise an exception for bad status codes
        
        tokens = token_response.json().get("tokens", [])
        near_balance = near_response.json().get("account", [{}])[0].get("amount", "0")
        
        # Process token balances
        token_balances = []
        for token in tokens:
            
            entry = [item for item in data if item.get('contract_address') == token["contract_id"]]

            if (len(entry) == 0):
                continue

            balance = (
                str(((Decimal(token["balance"]) + Decimal(near_balance) - Decimal(NEAR_BUFFER)) / Decimal(Decimal(10) ** int(entry[0]["decimals"]))))
                if token["contract_id"] == "wrap.near"
                else str((Decimal(token["balance"]) / Decimal(Decimal(10) ** int(entry[0]["decimals"]))))
            )
            
            if Decimal(balance) < 0:
                balance = "0"

            balance_usd = str((Decimal(entry[0]["price"]) * Decimal(balance)))
            
            if entry[0]["symbol"].upper() == "WNEAR":
                    entry[0]["symbol"] = "NEAR"
            
            if Decimal(balance) <= 0:
                continue
            
            token_balances.append({
                "contractId": entry[0]["defuse_asset_id"].replace("nep141:", ""),
                "symbol": entry[0]["symbol"],
                "blockchain": entry[0]["blockchain"],
                "balance": balance,
                "balance_usd": balance_usd
            })
        
        entry = [item for item in data if item.get('symbol') == "ZEC"]

        account = zcash.getAccountForAddress(env_vars["ZCASH_ADDRESS"])
        transparent_balance, shielded_balance = zcash.account_balance(account)
        zec_balance = Decimal(transparent_balance) + Decimal(shielded_balance) - Decimal("0.0004")
        if zec_balance < 0:
            zec_balance = 0
        
        balance_usd = str((Decimal(entry[0]["price"]) * Decimal(zec_balance)))
        token_balances.append({
                "contractId": "zec.omft.near",
                "symbol": f"{entry[0]['symbol']}",
                "blockchain": entry[0]["blockchain"],
                "balance": str(zec_balance),
                "balance_usd": balance_usd
            })


        if len(token_balances) == 0:
            return "You have no tokens in your wallet."
        
        
        return json.dumps(token_balances)
    
    except requests.RequestException as e:
        raise Exception(f"Request failed: {e}")
    except Exception as e:
        raise Exception(f"Internal server error: {e}")

async def _Intents_balance(account_id,data_old):
    data = (load_url("https://api-mng-console.chaindefuser.com/api/tokens"))
    if data["items"] is None:
        data = data_old
    else:
        data = data["items"]
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]
    token_ids = [item["defuse_asset_id"] for item in data]
    
    args = {
        "account_id": account_id,
        "token_ids": token_ids,
    }
    
    near = NearAccount(user_account_id,user_private_key)
    try:
        tr = await near.view("intents.near","mt_batch_balance_of",args)
        balance = {}
        balances = []
                
        for i in range(len(token_ids)):
            if Decimal(tr.result[i]) > 0:
                token = [item for item in data if item.get('defuse_asset_id') == token_ids[i]]
                
                if token[0]["symbol"].upper() == "WNEAR":
                    token[0]["symbol"] = "NEAR"
                
                
                prev = 0
                if token[0]["symbol"] in balance:
                    prev = Decimal(balance[token[0]["symbol"]]["amt"])
                    
                current = (Decimal(tr.result[i]) / Decimal(Decimal(10) ** int(token[0]["decimals"])))
                    
                balance[token[0]["symbol"]] = {
                    "amt" : str(prev + current),
                    "usd" : str(current * (Decimal(token[0]["price"])))
                }
        
        for tk in balance:
            
            balances.append({"TOKEN":tk,
                            "AMOUNT":balance[tk]["amt"],
                            "AMOUNT_IN_USD": balance[tk]["usd"]})
                
        return balances
    except Exception as e:
        raise Exception(f"Internal server error: {e}")
