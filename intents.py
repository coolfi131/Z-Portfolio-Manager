import base64
import hashlib
import json
import secrets
import time
from typing import Any, List, Optional, Union
from decimal import Decimal

import base58
import nacl.signing
import requests

# from py_near.account import Account
from borsh_construct import U32
from serializer import BinarySerializer
import zcash
from py_near.constants import DEFAULT_ATTACHED_GAS
from py_near.account import Account

default_mainnet_rpc = "https://rpc.mainnet.near.org"

import re

with open("env.json", "r") as file:
    env_vars = json.load(file)

with open("tokens.json", "r") as file:
    data = json.load(file)

INTENTS_CONTRACT = "intents.near"
url = "https://solver-relay-v2.chaindefuser.com/rpc"

headers = {
    "Content-Type": "application/json"
}

ED_PREFIX = "ed25519:"  

FT_DEPOSIT_GAS = 30000000000000
FT_TRANSFER_GAS = 50000000000000
FT_MINIMUM_STORAGE_BALANCE_LARGE = 1250000000000000000000


class NearAccount(Account):
    async def view(
        self,
        contract_id: str,
        method_name: str,
        args: dict,
        block_id: Optional[int] = None,
        threshold: Optional[int] = None,
        max_retries: int = 3,
    ):
        """Wrapper for the view method of the Account class, adding multiple retry attempts.

        Parameters
        ----------
        contract_id : str
            The ID of the contract to call.
        method_name : str
            The name of the method to invoke on the contract.
        args : dict
            The arguments to pass to the contract method.
        block_id : Optional[int]
            The block ID to query at.
        threshold : Optional[int]
            The threshold for the view function.
        max_retries : int
            The maximum number of retry attempts.

        Returns
        -------
        The result of the contract method call.

        Raises
        ------
        Exception
            If all retry attempts fail, the exception is propagated.

        """
        acc = Account(self.account_id, self.private_key, default_mainnet_rpc)
        await acc.startup()
        max_retries = min(max_retries, 10)

        for attempt in range(1, max_retries + 1):
            try:
                # Attempt to read the contract view method
                return await acc.view_function(contract_id, method_name, args, block_id, threshold)
            except Exception as e:
                # Log the error message for the current attempt
                print(
                    f"Attempt {attempt}/{max_retries} to view method '{method_name}' on contract "
                    f"'{contract_id}' failed with error: {e}"
                )

                # If it's the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise

    async def call(
        self,
        contract_id: str,
        method_name: str,
        args: dict,
        gas: int = DEFAULT_ATTACHED_GAS,
        amount: int = 0,
        nowait: bool = False,
        included: bool = False,
        max_retries: int = 1,
    ):
        """Wrapper for the call method of the Account class, adding multiple retry attempts.

        Parameters
        ----------
        contract_id : str
            The ID of the contract to call.
        method_name : str
            The name of the method to invoke on the contract.
        args : dict
            The arguments to pass to the contract method.
        gas : int
            The amount of gas to attach to the call.
        amount : int
            The amount of tokens to attach to the call.
        nowait : bool
            If nowait is True, return transaction hash, else wait execution.
        included : bool
            If included is True, return transaction hash, else wait execution
        max_retries : int
            The maximum number of retry attempts.

        Returns
        -------
        The result of the contract method call.

        Raises
        ------
        Exception
            If all retry attempts fail, the exception is propagated.

        """
        acc = Account(self.account_id, self.private_key, default_mainnet_rpc)
        await acc.startup()
        max_retries = min(max_retries, 10)

        for attempt in range(1, max_retries + 1):
            try:
                # Attempt to call the contract method
                return await acc.function_call(contract_id, method_name, args, gas, amount, nowait, included)
            except Exception as e:
                # Log the error message for the current attempt
                print(
                    f"Attempt {attempt}/{max_retries} to call method '{method_name}' on contract "
                    f"'{contract_id}' failed with error: {e}"
                )

                # If it's the last attempt, re-raise the exception
                if attempt == max_retries:
                    raise

    async def get_balance(self, account_id: Optional[str] = None) -> int:
        """Retrieves the balance of the specified NEAR account.

        Parameters
        ----------
        account_id : Optional[str]
            The ID of the account to retrieve the balance for. If not provided, the balance of the current
            account is retrieved.

        Returns
        -------
        int
            The balance of the specified account in yoctoNEAR.

        Raises
        ------
        Exception
            If there is an error retrieving the balance.

        """
        acc = Account(self.account_id, self.private_key, default_mainnet_rpc)
        await acc.startup()
        return await acc.get_balance(account_id)

    def __init__(
        self,
        account_id: Optional[str] = None,
        private_key: Optional[Union[List[Union[str, bytes]], str, bytes]] = None,
        rpc_addr: Optional[str] = None,
    ):
        self.account_id = account_id
        self.private_key = private_key
        super().__init__(account_id, private_key, rpc_addr)


async def add_public_key(public_key):
    # Setup
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]
    near = NearAccount(user_account_id, user_private_key)

    has_public_key = await near.view(
        "intents.near",
        "has_public_key",
        {
            "account_id": user_account_id,
            "public_key": str(public_key),
        }
    )

    if has_public_key:
        return

    # Add the public_key
    result = await near.call(
        "intents.near",
        "add_public_key",
        {"public_key": str(public_key)},
        FT_DEPOSIT_GAS,  # optional: you can specify gas
        1  # 1 yoctoNEAR
    )


async def get_withdraw_message_to_sign(signer_id, token, receiver_id, amount, blockchain):
    # now + 3 min in a format of 2025-01-21T14:55:40.323Z
    exp_time = (time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() + 180)))
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]

    near = NearAccount(user_account_id, user_private_key)

    nep141balance = await near.view(
        contract_id="wrap.near",
        method_name="storage_balance_of",
        args={
            "account_id": user_account_id
        }
    )

    if nep141balance.result:
        nep141balance = int(nep141balance.result['available'])
    else:
        nep141balance = 0

    storage_deposit = 0 if nep141balance > FT_MINIMUM_STORAGE_BALANCE_LARGE else FT_MINIMUM_STORAGE_BALANCE_LARGE

    message = "dummy"

    if token == "wrap.near":
        message = {
            "signer_id": signer_id,
            "deadline": exp_time,
            "intents": [
                {
                    "intent": "native_withdraw" ,
                    "receiver_id": receiver_id,
                    "amount": str(amount)
                }
            ]
        }
    elif blockchain == "near":
        message = {
            "signer_id": signer_id,
            "deadline": exp_time,
            "intents": [
                {
                    "intent": "ft_withdraw" ,
                    "receiver_id": receiver_id,
                    "token": token,
                    "amount": str(amount),
                    "deposit": str(storage_deposit)
                }
            ]
        }
    else:
        message = {
            "signer_id": signer_id,
            "deadline": exp_time,
            "intents": [
                {
                    "intent": "ft_withdraw",
                    "receiver_id": token,
                    "amount": str(amount),
                    "token": token,
                    "deposit": str(storage_deposit),
                    "memo": f"WITHDRAW_TO:{receiver_id}"
                }
            ]
        }
    
    # Convert message dictionary to JSON string
    message_str = json.dumps(message)
    return message_str


def get_swap_message_to_sign(signer_id, token_in, amount_in, token_out, amount_out, exp_time):
    # Construct the message dictionary
    message = {
        "signer_id": signer_id,
        "deadline": exp_time,
        "intents": [
            {
                "intent": "token_diff",
                "diff": {
                    f"{token_in}": f"-{amount_in}",
                    f"{token_out}": amount_out
                }
            }
        ]
    }

    message_str = json.dumps(message)
    return message_str


def generate_nonce():
    random_array = secrets.token_bytes(32)
    return base64.b64encode(random_array).decode('utf-8')


def base64_to_uint8array(base64_string):
    binary_data = base64.b64decode(base64_string)
    return list(binary_data)


def convert_nonce(value: Union[str, bytes, list[int]]):
    """Converts a given value to a 32-byte nonce."""
    if isinstance(value, bytes):
        if len(value) > 32:
            raise ValueError("Invalid nonce length")
        if len(value) < 32:
            value = value.rjust(32, b"0")
        return value
    elif isinstance(value, str):
        nonce_bytes = value.encode("utf-8")
        if len(nonce_bytes) > 32:
            raise ValueError("Invalid nonce length")
        if len(nonce_bytes) < 32:
            nonce_bytes = nonce_bytes.rjust(32, b"0")
        return nonce_bytes
    elif isinstance(value, list):
        if len(value) != 32:
            raise ValueError("Invalid nonce length")
        return bytes(value)
    else:
        raise ValueError("Invalid nonce format")


class Payload:
    def __init__(  # noqa: D107
            self, message: str, nonce: Union[bytes, str, List[int]], recipient: str, callback_url: Optional[str] = None
    ):
        self.message = message
        self.nonce = convert_nonce(nonce)
        self.recipient = recipient
        self.callbackUrl = callback_url


PAYLOAD_SCHEMA: list[list[Any]] = [
    [
        Payload,
        {
            "kind": "struct",
            "fields": [
                ["message", "string"],
                ["nonce", [32]],
                ["recipient", "string"],
                [
                    "callbackUrl",
                    {
                        "kind": "option",
                        "type": "string",
                    },
                ],
            ],
        },
    ]
]

def serialize_intent(intent_message, recipient, nonce):
    payload2 = Payload(intent_message, nonce, recipient, None)
    borsh_payload = BinarySerializer(dict(PAYLOAD_SCHEMA)).serialize(payload2)

    base_int = 2 ** 31 + 413
    base_int_serialized = U32.build(base_int)
    combined_data = base_int_serialized + borsh_payload
    hash_result = hashlib.sha256(combined_data).digest()
    return hash_result

def getAddressChains(address):
    valid_chains = []
    
    if re.match(r'^(([a-z\d]+[-_])*[a-z\d]+\.)*([a-z\d]+[-_])*[a-z\d]+$', address):
        valid_chains.append("near")
    
    if re.match(r'^0x[a-fA-F0-9]{40}$', address):
        valid_chains.extend(["eth", "base", "arb", "gnosis", "bera"])
    
    if (re.match(r'^1[1-9A-HJ-NP-Za-km-z]{25,34}$', address) or
        re.match(r'^3[1-9A-HJ-NP-Za-km-z]{25,34}$', address) or
        re.match(r'^bc1[02-9ac-hj-np-z]{11,87}$', address) or
        re.match(r'^bc1p[02-9ac-hj-np-z]{42,87}$', address)):
        valid_chains.append("btc")
    
    # try:
    #     if PublicKey(address).is_on_curve():
    #         valid_chains.append("sol")
    # except:
    #     pass
    
    if re.match(r'^[DA][1-9A-HJ-NP-Za-km-z]{25,33}$', address):
        valid_chains.append("doge")
    
    # if xrp_isValidClassicAddress(address) or xrp_isValidXAddress(address):
    #     valid_chains.append("xrp")
    
    if zcash.validate_zcash_address(address)["isvalid"]:
        valid_chains.append("zec")
    
    return valid_chains

async def withdraw_from_intents(token, amount, receiver_id, data):
    
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]
    near = NearAccount(user_account_id, user_private_key)

    valid_chains = getAddressChains(receiver_id)

    if not valid_chains:
        return False

    match = [obj for obj in data if obj["symbol"] == token.upper() and obj["blockchain"] in valid_chains]
    
    if not match:
      return False

    token_data = match[0]
    amount_out_hr = amount
    amount = int(Decimal(amount) * Decimal(10) ** int(token_data["decimals"]))

    if amount < int(token_data["min_withdraw_amount"]):
        return False

    contract_id = token_data["defuse_asset_id"].replace("nep141:", "")


    token_list = [obj for obj in data if obj["symbol"] == token.upper()]
    
    if len(token_list) > 1:
        contract_list = [obj["defuse_asset_id"] for obj in token_list]
        tr = await near.view("intents.near", "mt_batch_balance_of",
                    {
                        "account_id": user_account_id,
                        "token_ids": contract_list,
                    })
        result = tr.result
        i = 0
        j = contract_list.index(token_data["defuse_asset_id"])
        result[j] = Decimal(result[j]) / (Decimal(10) ** int(token_data["decimals"]))

        for i, token_obj in enumerate(token_list):
            if i == j or Decimal(result[i]) == 0:
                continue
            result[i] = Decimal(result[i]) / (Decimal(10) ** int(token_obj["decimals"]))
            if result[j] >= Decimal(amount):
                break

            amount_swapped = await _intent_swap(token_obj["symbol"], token_data["symbol"], result[i], data, token_obj["defuse_asset_id"], token_data["defuse_asset_id"])
            
            result[i] = 0
            result[j] = result[j] + amount_swapped

            i = i + 1
            
    near = NearAccount(user_account_id)
    args = {
        "account_id": user_account_id,
        "token_ids": [token_data["defuse_asset_id"]],
    }
    
    tr = await near.view("intents.near", "mt_batch_balance_of", args)
    if Decimal(amount) > Decimal(tr.result[0]):
        print("Amount is more than the maximum available amount to withdraw. Withdrawing the complete amount")
        amount = Decimal(tr.result[0])
    
    message_str = await get_withdraw_message_to_sign(user_account_id, contract_id, receiver_id, amount, token_data["blockchain"])
    nonce = generate_nonce()
    
    nonce_uint8array = base64_to_uint8array(nonce)
    quote_hash_solver = serialize_intent(message_str, INTENTS_CONTRACT, nonce_uint8array)

    private_key_base58 = user_private_key[len(ED_PREFIX):]
    private_key_bytes = base58.b58decode(private_key_base58)

    if len(private_key_bytes) != 64:
        raise ValueError("The private key must be exactly 64 bytes long")

    private_key_seed = private_key_bytes[:32]
    signing_key = nacl.signing.SigningKey(private_key_seed)
    public_key = signing_key.verify_key
    signed = signing_key.sign(quote_hash_solver)
    
    final_signature = base58.b58encode(signed.signature).decode("utf-8")
    public_key_base58 = base58.b58encode(public_key.encode()).decode("utf-8")

    request = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "publish_intent",
        "params": [
            {
                "quote_hashes": [],
                "signed_data": {
                    "payload": {
                        "message": message_str,
                        "nonce": nonce,
                        "recipient": INTENTS_CONTRACT,
                    },
                    "standard": "nep413",
                    "signature": f"ed25519:{final_signature}",
                    "public_key": f"ed25519:{base58.b58encode(public_key.encode()).decode()}",
                }
            }
        ]
    }

    response = requests.post(url, headers=headers, json=request)
    response.raise_for_status()
    resp = response.json()

    if resp["result"]["status"] == "OK":
        intent_hash = resp["result"]["intent_hash"]

        settled, result = get_intent_settled_status(intent_hash)
        if settled:
            transaction_hash = result["result"]["data"]["hash"]
            print("Transaction Hash:", transaction_hash)
            return True

        else:
            return None

    else:
        return None

async def intent_swap(token_in, token_out, amount_in, token_data, contract_in = "", contract_out = ""):
    
    token_list = [obj for obj in data if obj["symbol"] == token_in.upper()]
    
    matches_in = [obj for obj in token_data if obj["symbol"] == token_in.upper()]
    
    if not matches_in:
      return False
  
    token_data_in = matches_in[0] if not contract_in else next((obj for obj in matches_in if obj["defuse_asset_id"] == contract_in), None)
    
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]
    
    near = NearAccount(user_account_id, user_private_key)
    
    if len(token_list) > 1:
        contract_list = [obj["defuse_asset_id"] for obj in token_list]
        tr = await near.view("intents.near", "mt_batch_balance_of",
                    {
                        "account_id": user_account_id,
                        "token_ids": contract_list,
                    })
        result = tr.result
        i = 0
        j = contract_list.index(token_data_in["defuse_asset_id"])
        result[j] = Decimal(result[j]) / (Decimal(10) ** int(token_data_in["decimals"]))

        for i, token_obj in enumerate(token_list):
            if i == j or Decimal(result[i]) == 0:
                continue
            result[i] = Decimal(result[i]) / (Decimal(10) ** int(token_obj["decimals"]))
            if result[j] >= Decimal(amount_in):
                break

            amount_swapped = await _intent_swap(token_obj["symbol"], token_data_in["symbol"], result[i], data, token_obj["defuse_asset_id"], token_data_in["defuse_asset_id"])
            
            result[i] = 0
            result[j] = result[j] + amount_swapped

            i = i + 1
            
    return await _intent_swap(token_in, token_out, amount_in, token_data, contract_in, contract_out)
    

async def _intent_swap(token_in, token_out, amount_in, token_data, contract_in = "", contract_out = ""):
    
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]
    
    matches_in = [obj for obj in token_data if obj["symbol"] == token_in.upper()]
    
    if not matches_in:
      return False
  
    matches_out = [obj for obj in token_data if obj["symbol"] == token_out.upper()]
    
    if not matches_out:
      return False
  
    token_data_in = matches_in[0] if not contract_in else next((obj for obj in matches_in if obj["defuse_asset_id"] == contract_in), None)

    if not token_data_in:
        return False
    
    token_data_out = matches_out[0] if not contract_out else next((obj for obj in matches_out if obj["defuse_asset_id"] == contract_out), None)

    if not token_data_out:
        return False
    
    amount = int(Decimal(amount_in) * Decimal(10) ** int(token_data_in["decimals"]))
    
    near = NearAccount(user_account_id)
    args = {
        "account_id": user_account_id,
        "token_ids": [token_data_in["defuse_asset_id"]],
    }
    
    tr = await near.view("intents.near", "mt_batch_balance_of", args)
    if Decimal(amount) > Decimal(tr.result[0]):
        print("Amount is more than the maximum available amount to swap. Swapping the complete amount")
        amount = Decimal(tr.result[0])
    
    data = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "quote",
        "params": [
            {
                "defuse_asset_identifier_in": token_data_in["defuse_asset_id"],
                "defuse_asset_identifier_out": token_data_out["defuse_asset_id"],
                "exact_amount_in": str(amount)
            }
        ]
    }

    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, data=json.dumps(data))

        try:
            response.raise_for_status()
            parsed_response = response.json()

            if parsed_response.get('result') is not None:
                break
            else:
                print(f"Empty result on attempt {attempt + 1}. Retrying in {retry_delay} second(s)...")
                time.sleep(retry_delay)

        except Exception as err:
            print("HTTP error occurred:", err)
            print("Error details:", response.text)
            return None

    else:
        print("Error: result is not provided")
        return None

    amount_out = 0

    quote_hash = None
    amount_in = None
    expiration_time = None

    if "result" in parsed_response and len(parsed_response["result"]) > 0:
        for result in parsed_response["result"]:
            # find a quote with the highest amount_out
            if int(result["amount_out"]) > int(amount_out):
                _asset_in = result["defuse_asset_identifier_in"]
                _asset_out = result["defuse_asset_identifier_out"]
                amount_in = result["amount_in"]
                quote_hash = result["quote_hash"]
                amount_out = result["amount_out"]
                expiration_time = result["expiration_time"]
            else:
                this_amount_out = result["amount_out"]
    else:
        print("Error:", response.status_code, response.text)
        return False

    if not amount_in or not expiration_time or not quote_hash:
        print("Error with quote data:", response.status_code, response.text)
        return False

    message_str = get_swap_message_to_sign(user_account_id, token_data_in["defuse_asset_id"], amount_in, token_data_out["defuse_asset_id"],
                                           amount_out, expiration_time)
    nonce = generate_nonce()

    quote_hashes = [quote_hash]

    nonce_uint8array = base64_to_uint8array(nonce)

    quote_hash_solver = serialize_intent(message_str, INTENTS_CONTRACT, nonce_uint8array)

    private_key_base58 = user_private_key[len(ED_PREFIX):]
    private_key_bytes = base58.b58decode(private_key_base58)

    if len(private_key_bytes) != 64:
        raise ValueError("The private key must be exactly 64 bytes long")

    private_key_seed = private_key_bytes[:32]
    signing_key = nacl.signing.SigningKey(private_key_seed)
    public_key = signing_key.verify_key
    signed = signing_key.sign(quote_hash_solver)
    _signature = base64.b64encode(signed.signature).decode("utf-8")

    final_signature = base58.b58encode(signed.signature).decode("utf-8")

    public_key_base58 = base58.b58encode(public_key.encode()).decode("utf-8")
    _full_public_key = ED_PREFIX + public_key_base58

    await add_public_key(_full_public_key)

    request = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "publish_intent",
        "params": [
            {
                "quote_hashes": quote_hashes,
                "signed_data": {
                    "payload": {
                        "message": message_str,
                        "nonce": nonce,
                        "recipient": INTENTS_CONTRACT,
                    },
                    "standard": "nep413",
                    "signature": f"ed25519:{final_signature}",
                    "public_key": f"ed25519:{base58.b58encode(public_key.encode()).decode()}",
                }
            }
        ]
    }

    intent_response, settled, intent_hash, amount_in_usd, amount_out_usd, result = (
        make_intent_swap(request, token_data_out["symbol"], amount_in, token_data_in["decimals"], amount_out, token_data_out["decimals"]))

    if not settled:
        # Try again
        time.sleep(2)
        intent_response, settled, intent_hash, amount_in_usd, amount_out_usd, result = (
            make_intent_swap(request, token_data_out["symbol"], amount_in, token_data_in["decimals"], amount_out, token_data_out["decimals"]))

        if not settled:
            # Try again
            time.sleep(10)
            intent_response, settled, intent_hash, amount_in_usd, amount_out_usd, result = (
                make_intent_swap(request, token_data_out["symbol"], amount_in, token_data_in["decimals"], amount_out, token_data_out["decimals"]))

    if settled:
        transaction_hash = result["result"]["data"]["hash"]
        amount_out = Decimal(amount_out) / (Decimal(10) ** int(token_data_out["decimals"]))
        amount_in = Decimal(amount_in) / (Decimal(10) ** int(token_data_in["decimals"]))
        print("Transaction Hash:", transaction_hash)
        return amount_out

    else:
        return False

def make_intent_swap(request, symbol_out, amount_in, token_in_decimals, amount_out, token_out_decimals):

    response = requests.post(url, headers=headers, json=request)
    response.raise_for_status()
    resp = response.json()

    amount_in_usd = f"{float(amount_in) / pow(10, token_in_decimals):.5f}"
    amount_out_usd = f"{float(amount_out) / pow(10, token_out_decimals):.5f}"

    if resp["result"]["status"] == "OK":
        intent_hash = resp["result"]["intent_hash"]

        settled, result = get_intent_settled_status(intent_hash)

        return resp, settled, intent_hash, amount_in_usd, amount_out_usd, result

    else:
        return resp, False, False, amount_in_usd, amount_out_usd, resp

def get_intent_settled_status(intent_hash):
    data = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "get_status",
        "params": [
            {
                "intent_hash": intent_hash
            }
        ]
    }

    start_time = time.time()
    status = "GOOD"
    while True:
        time.sleep(0.2)

        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        resp = response.json()

        if resp['result']['status'] == "SETTLED":
            return True, resp

        elif (resp['result']['status'] == "NOT_FOUND_OR_NOT_VALID_ANYMORE"
            or resp['result']['status'] == "NOT_FOUND_OR_NOT_VALID"):
            print("Intent not found or not valid anymore")
            return False, resp
        
        elif resp['result']['status'] == "FAILED":
            return False, resp

        elif time.time() - start_time > 30:
            print("Timeout: Operation took longer than 30 seconds")
            return False, resp

        if status != resp['result']['status']:
            status = resp['result']['status']
        
        
async def _deposit_to_intents(data, amount, sender, token_symbol = ""):
    
    supported_data = data
    user_account_id = env_vars["ACCOUNT_ID"]
    user_private_key = env_vars["PRIVATE_KEY"]
    matches = [obj for obj in supported_data if obj["symbol"] == token_symbol.upper() and obj["blockchain"] in ("near", "zec")]
    
    if not matches:
      return False
  
    token = matches[0]
    if token["symbol"] == "ZEC":
        txid = await zcash.deposit(sender, amount)
        return True
    
    amount = Decimal(amount) * Decimal(10) ** int(token["decimals"])
    amount = int(amount) 
    contract_id = token["defuse_asset_id"].replace("nep141:", "")

    near = NearAccount(user_account_id, user_private_key)

    nep141balance = await near.view(
        contract_id="wrap.near",
        method_name="storage_balance_of",
        args={
            "account_id": user_account_id
        }
    )
    
    if nep141balance.result:
        nep141balance = int(nep141balance.result['available'])
    else:
        nep141balance = 0

    storage_payment = FT_MINIMUM_STORAGE_BALANCE_LARGE - nep141balance

    if contract_id == "wrap.near":

        token_response = requests.get(f"https://api.fastnear.com/v1/account/{user_account_id}/ft")
        token_response.raise_for_status()

        tokens = token_response.json().get("tokens", [])
        near_balance = int(next((token["balance"] for token in tokens if token["contract_id"] == "wrap.near"), 0))

        near_amount = 0 if amount - near_balance < 0 else (amount) - near_balance

        if (storage_payment > 0 or near_amount > 0):
            tr = await near.call(contract_id, "near_deposit", {}, FT_DEPOSIT_GAS, storage_payment + near_amount)
            if "SuccessValue" not in tr.status:
                return False
            
        tr = await near.call(contract_id, "ft_transfer_call",
                {"receiver_id": INTENTS_CONTRACT, "amount": str(amount), "msg": ""},
                FT_TRANSFER_GAS,
                1)
        
        if "SuccessValue" not in tr.status:
                return False
    
    else:
      if storage_payment > 0:
        tr = await near.call(contract_id, "storage_deposit",
                {
                  "account_id": INTENTS_CONTRACT,
                #   "registration_only": True,
                },
                FT_DEPOSIT_GAS, storage_payment)
        
        if "SuccessValue" not in tr.status:
            return False
        
      tr = await near.call(contract_id, "ft_transfer_call",
            {"receiver_id": INTENTS_CONTRACT, "amount": str(amount), "msg": ""},
              FT_TRANSFER_GAS,
            1)
      
      if "SuccessValue" not in tr.status:
            return False

    amount = float(amount) / float(Decimal(10) ** int(token["decimals"]))
    print("Transaction Hash:", tr.transaction.hash)
    return True
