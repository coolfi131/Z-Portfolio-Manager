from decimal import Decimal
import time
import requests
from requests.auth import HTTPBasicAuth
import intents
from intents import NearAccount
import json

rpc_url = "https://bridge.chaindefuser.com/rpc"
zcash_fees = Decimal("0.0002")
zcash_account = None


with open("env.json", "r") as file:
    env_vars = json.load(file)
    
with open("tokens.json", "r") as file:
    data = json.load(file)

def createAccount():
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_getnewaccount",
        "params": []
    }

    node_url = env_vars["ZCASH_NODE_URL"]
    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()

    if response["result"]["account"]:
        return int(response["result"]["account"])
    
    return -1

def getAddressForAccount(account):
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_listaccounts",
        "params": []
    }

    node_url = env_vars["ZCASH_NODE_URL"]
    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
    if response["result"][int(account)]["addresses"]:
        return response["result"][int(account)]["addresses"][0]["ua"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_getaddressforaccount",
        "params": [int(account)]
    }

    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()

    if response["result"]["address"]:
        return response["result"]["address"]
    else:
        return ""

def getAccountForAddress(address):
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "listaddresses",
        "params": []
    }

    try:
        data = requests.post(env_vars["ZCASH_NODE_URL"], json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
        
        if "result" not in data:
            raise ValueError("Invalid response: missing 'result' key")
        
        list_addresses = data["result"]
        
        for wallet in list_addresses:
            if "unified" not in wallet:
                continue
            for account_info in wallet["unified"]:
                if "addresses" not in account_info or not isinstance(account_info["addresses"], list):
                    continue
                for addr in account_info["addresses"]:
                    if "address" in addr and addr["address"] == address:
                        return account_info["account"]
        
        return None  # Address not found
    
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return None

def getZcashIntentAccount():
    try:
        # Open the file in read mode first to check existing content
        with open(env_vars["ZCASH_ACCOUNT_FILE"], "r") as file:
            account = file.read().strip()
    except FileNotFoundError:
        account = -1

    # If account is empty or -1, create a new account
    if account == -1:
        account = createAccount()

    # Validate the account 
    try:
        zcash_account = account
    except ValueError:
        return -1

    # Check if account creation failed
    if zcash_account == -1:
        return -1

    # Open file in write mode to update the account
    with open(env_vars["ZCASH_ACCOUNT_FILE"], "w") as file:
        file.write(str(account))

    # Update environment variables

    return zcash_account

def validate_zcash_address(address):
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_validateaddress",
        "params": [
            address
        ]
    }

    node_url = env_vars["ZCASH_NODE_URL"]
    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
    if not response["result"]["isvalid"]:
        return {"isvalid": response["result"]["isvalid"], "address_type": "invalid"}
    return {"isvalid": response["result"]["isvalid"], "address_type": response["result"]["address_type"]}

def wallet_balance():
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "getwalletinfo",
        "params": []
    }

    node_url = env_vars["ZCASH_NODE_URL"]
    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
    return response["result"]["balance"], response["result"]["shielded_balance"]

def account_balance(account):
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    token_data = [obj for obj in data if obj["symbol"] == 'ZEC'][0]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_getbalanceforaccount",
        "params": [int(account)]
    }

    response = requests.post(env_vars["ZCASH_NODE_URL"], json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()

    balance_transparent = 0
    balance_shielded = 0

    if response["result"]:
        pools = response["result"]["pools"]

        if pools and "transparent" in pools and pools["transparent"]["valueZat"]:
            balance_transparent = Decimal(pools["transparent"]["valueZat"]) / (Decimal(10) ** int(token_data["decimals"]))

        if pools and "sapling" in pools and pools["sapling"]["valueZat"]:
            balance_shielded = Decimal(pools["sapling"]["valueZat"]) / (Decimal(10) ** int(token_data["decimals"]))

        if pools and "orchard" in pools and pools["orchard"]["valueZat"]:
            balance_shielded = balance_shielded + Decimal(pools["orchard"]["valueZat"]) / (Decimal(10) ** int(token_data["decimals"]))

    return balance_transparent, balance_shielded



def transfer(sender, amount, recipient, args = [1, str(zcash_fees), 'NoPrivacy']):
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]

    headers = {"Content-Type": "text/plain"}
    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_sendmany",
        "params": [
            sender,
            [
                {
                    "address": recipient,
                    "amount": str(Decimal(amount) - zcash_fees)
                }
            ],
        ]
    }

    payload["params"].extend(args)

    node_url = env_vars["ZCASH_NODE_URL"]
    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
    if not response["result"]:
        return False
    opid = response["result"]

    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_listoperationids",
        "params": []
    }

    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
    
    if opid not in response["result"]:
        return opid

    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_getoperationstatus",
        "params": [
            [opid]
        ]
    }

    start_time = time.time()
    timeout = 300
    
    while True:
        response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
        if response["result"] and response["result"][0]:  # Check if result is available
            result = response["result"][0]
            if result["status"] == "success":
                txid = result["result"]["txid"]
                return txid
            elif result["status"] == "failed":
                print(result)
                return None
        
        if time.time() - start_time > timeout:  # Check if 2 minutes have passed
            print("[red]Timeout: Operation did not complete within 2 minutes[/red]")
            return None  # Or handle timeout case accordingly
        
        time.sleep(2)
        

            

async def deposit(sender, amount):
    
    user_account_id = env_vars["ACCOUNT_ID"]
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]
    headers = {"Content-Type": "text/plain"}

    account = getAccountForAddress(sender)
    balance_transparent, balance_shielded = account_balance(account)

    match = [obj for obj in data if obj["symbol"] == "ZEC"]
    token_data = match[0]

    amount = Decimal(amount) + Decimal(zcash_fees)
    if Decimal(amount) > Decimal(balance_shielded) + Decimal(balance_transparent):
        print(f"You have insufficiant balance of {Decimal(balance_shielded) + Decimal(balance_transparent)}. Cannot deposit {amount}")
        return False

    if Decimal(amount) > Decimal(balance_shielded) and Decimal(amount) < Decimal(balance_shielded) + Decimal(balance_transparent):
        args = [
            1,
            str(zcash_fees),
            "AllowRevealedSenders"
        ]

        amount = Decimal(amount) + Decimal(zcash_fees)
        txid = transfer(sender, amount, sender, args)
        if not txid:
            return False
        
        print("Transaction Id:", txid)
        
        start_time = time.time()
        timeout = 300
        while True:
            _, shielded = account_balance(account)
            if Decimal(shielded) > Decimal(amount):
                break
            
            if time.time() - start_time > timeout:  # Check if 5 minutes have passed
                print("[red]Timeout: Operation did not complete within 5 minutes[/red]")
                return None

            time.sleep(2)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "deposit_address",
        "params": [{
            "account_id": user_account_id,
            "chain": "zec:mainnet"
        }]
    }
    
    # wait unitl the amount gets confirmed on intents
    args = {
        "account_id": user_account_id,
        "token_ids": ["nep141:zec.omft.near"],
    }

    response = requests.post(rpc_url, json=payload).json()
    deposit_address = response["result"]["address"]

    args = [
        1,
        str(zcash_fees),
        "NoPrivacy"
    ]

    txid = transfer(sender, amount, deposit_address, args)
    print("Transaction Id:", txid)
    
    start_time = time.time()
    timeout = 600
    
    near = NearAccount(user_account_id)
    args = {
        "account_id": user_account_id,
        "token_ids": ["nep141:zec.omft.near"],
    }
    
    while True:
        tr = await near.view("intents.near", "mt_batch_balance_of", args)
        zec_balance = Decimal(tr.result[0]) / Decimal(Decimal(10) ** int(token_data["decimals"]))
        
        if Decimal(zec_balance) >= Decimal(amount) - Decimal(zcash_fees):
            break
        
        if time.time() - start_time > timeout:  # Check if 5 minutes have passed
            return txid
        
        time.sleep(2)
        
    return txid

async def withdraw(token, amount, recipient, data):
    username = env_vars["ZCASH_USER"]
    password = env_vars["ZCASH_PASS"]
    headers = {"Content-Type": "text/plain"}
    
    obj = validate_zcash_address(recipient)
    is_valid, address_type = obj["isvalid"], obj["address_type"]
    if not is_valid:
        print(f"Address {recipient} is not valid for zcash chain.")
        return False

    match = [obj for obj in data if obj["symbol"] == token.upper()]

    if not match:
      print(f"Token {token} may not be supported for this app.")
      return False

    token_data = match[0]

    if address_type in ("p2pkh", "p2sh"):
        return await intents.withdraw_from_intents(token, amount, recipient, data)
    
    account = getZcashIntentAccount()
    if account == -1:
        return False
    
    
    unified_address = getAddressForAccount(account)

    if not unified_address:
        return False

    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_listunifiedreceivers",
        "params": [unified_address]
    }

    node_url = env_vars["ZCASH_NODE_URL"]
    response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()

    transparent_address = response["result"]["p2pkh"] or response["result"]["p2sh"]
    shielded_address = response["result"]["sapling"] or response["result"]["orchard"]

    result = await intents.withdraw_from_intents(token, amount, transparent_address, data)
    if not result:
        return False
    
    payload = {
        "jsonrpc": "2.0",
        "id": "dontcare",
        "method": "withdrawal_status",
        "params": [{
            "withdrawal_hash": result 
        }]
    }
    
    start_time = time.time()
    timeout = 300
    hash = None
    to_print = True
    to_break = 3
    
    while True:
        response = requests.post(rpc_url, json=payload).json()
        
        if "result" in response:
            res = response["result"]
            
            if "withdrawals" in res:
                
                withdrawals = res["withdrawals"][0]
                hash = withdrawals["data"]["transfer_tx_hash"]
                status = withdrawals["status"]

                if status != "PENDING":
                    break
                
                if to_print:
                    print("Transaction Hash:", hash)
                    to_print = False
                    
        else:
            if to_break < 0:
                break
            to_break = to_break - 1
            time.sleep(5)

        if time.time() - start_time > timeout:  # Check if 5 minutes have passed
            print("[red]Timeout: Operation did not complete within 5 minutes[/red]")
            return None
    
        time.sleep(2)


    payload = {
        "jsonrpc": "1.0",
        "id": "curltest",
        "method": "z_getbalanceforaccount",
        "params": [int(account)]
    }

    start_time = time.time()
    timeout = 300

    while True:
        response = requests.post(node_url, json=payload, headers=headers, auth=HTTPBasicAuth(username, password)).json()
        if response["result"]:
            pools = response["result"]["pools"]

            if pools and pools["transparent"] and pools["transparent"]["valueZat"]:
                balance = Decimal(pools["transparent"]["valueZat"]) / (Decimal(10) ** int(token_data["decimals"]))
                if Decimal(amount) - zcash_fees <= balance:
                    break

        if time.time() - start_time > timeout:  # Check if 5 minutes have passed
            print("[red]Timeout: Operation did not complete within 5 minutes[/red]")
            return None

        time.sleep(2)

    args = [
        1,
        str(zcash_fees),
        "AllowRevealedSenders"
    ]

    txid = transfer(unified_address, amount, recipient, args)
    print("Transaction Hash:", txid)
    return txid