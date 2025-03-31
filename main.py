import asyncio
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

import requests
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.progress import Progress

from portfolio import PortfolioManager
import intents
from decimal import Decimal
import zcash

# Create a global console object
console = Console()

with open("env.json", "r") as file:
    env_vars = json.load(file)
    if not env_vars["ACCOUNT_ID"] or not env_vars["PRIVATE_KEY"] or not env_vars["ZCASH_NODE_URL"]:
        console.print("[bold red]Error: ACCOUNT_ID, PRIVATE_KEY, and ZCASH_NODE_URL must be set in env file[/bold red]")
        sys.exit(1)
        
    if not env_vars["ZCASH_USER"] or not env_vars["ZCASH_PASS"] or not env_vars["ZCASH_ACCOUNT_FILE"]:
        console.print("[bold red]Error: ZCASH_USER, ZCASH_PASS, and ZCASH_ACCOUNT_FILE must be set in env file[/bold red]")
        sys.exit(1)
        
    if not env_vars["ZCASH_ADDRESS"] or not env_vars["ZCASH_ACCOUNT_FILE"]:
        console.print("[bold red]Error: ZCASH_ADDRESS must be set in env file[/bold red]")
        sys.exit(1)

async def update_token_prices(tokens_data):
    """Update token prices from CoinGecko API."""
    
    try:
        # Get the list of token symbols
        token_ids = {}
        for token in tokens_data:
            symbol = token["symbol"].lower()
            # Map symbols to CoinGecko IDs (simplified mapping)
            if symbol == "near":
                token_ids[token["symbol"]] = "near"
            elif symbol == "eth":
                token_ids[token["symbol"]] = "ethereum"
            elif symbol == "btc":
                token_ids[token["symbol"]] = "bitcoin"
            elif symbol == "usdc":
                token_ids[token["symbol"]] = "usd-coin"
            elif symbol == "sol":
                token_ids[token["symbol"]] = "solana"
            elif symbol == "doge":
                token_ids[token["symbol"]] = "dogecoin"
            elif symbol == "xrp":
                token_ids[token["symbol"]] = "ripple"
            elif symbol == "zec":
                token_ids[token["symbol"]] = "zcash"
        
        # Make API call to CoinGecko
        ids_string = ",".join(token_ids.values())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_string}&vs_currencies=usd"
        response = requests.get(url)
        
        if response.status_code != 200:
            console.print(f"[bold red]Error fetching prices: {response.status_code}[/bold red]")
            return tokens_data
        
        prices = response.json()
        
        # Update token data with prices
        for token in tokens_data:
            symbol = token["symbol"]
            if symbol in token_ids and token_ids[symbol] in prices:
                token["price"] = prices[token_ids[symbol]]["usd"]
            else:
                console.print(f"[bold yellow]Warning: No price found for {symbol}[/bold yellow]")
                token["price"] = 0  # Default price
        
        return tokens_data
    
    except Exception as e:
        console.print(f"[bold red]Error updating token prices: {str(e)}[/bold red]")
        return tokens_data

def load_metadata():
    """Load metadata from metadata.json"""
    try:
        with open("metadata.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return None

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading config: {str(e)}[/bold red]")
        return None

def load_tokens():
    """Load token data from tokens.json"""
    try:
        with open("tokens.json", "r") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading tokens: {str(e)}[/bold red]")
        return None

async def initialize_portfolio_manager():
    """Initialize the PortfolioManager"""
    console.print(Panel("[bold blue]Initializing Portfolio Manager[/bold blue]", expand=False))
    
    # Load configuration
    
    
    config = load_config()
    if not config:
        return None
    
    # Load tokens
    tokens_data = load_tokens()
    if not tokens_data:
        return None
    
    # Get account ID and private key from environment variables
    account_id = env_vars["ACCOUNT_ID"]
    private_key = env_vars["PRIVATE_KEY"]
    zec_account_id = env_vars["ZCASH_ADDRESS"]
    
    if not account_id or not private_key:
        console.print("[bold red]Error: ACCOUNT_ID and PRIVATE_KEY environment variables must be set[/bold red]")
        console.print("Please set them with: export ACCOUNT_ID=your_account_id export PRIVATE_KEY=your_private_key")
        return None
    
    # Initialize portfolio manager
    try:
        console.print(f"Initializing portfolio manager for account: {account_id}")
        
        portfolio_manager = PortfolioManager(
            account_id=account_id,
            zec_account_id=zec_account_id,
            target_allocation=config["target_allocation"],
            rebalance_threshold=config["rebalance_threshold"],
            stop_loss=config.get("stop_loss"),
            take_profit=config.get("take_profit"),
            token_data=tokens_data
        )
        
        console.print("[bold green]Portfolio manager initialized successfully![/bold green]")
        return portfolio_manager
    
    except Exception as e:
        console.print(f"[bold red]Error initializing portfolio manager: {str(e)}[/bold red]")
        return None


async def view_performance(portfolio_manager):
    """View detailed portfolio performance metrics"""
    console.print(Panel("[bold blue]Portfolio Performance[/bold blue]", expand=False))
    
    with console.status("[bold green]Analyzing portfolio performance...[/bold green]"):
        
        wallet_balance = portfolio_manager.wallet_balance 
        intents_balance = portfolio_manager.intents_balance
        total_value = portfolio_manager.current_portfolio_value
        
        # Get current allocation
        current_allocation = await portfolio_manager.calculate_current_allocation()
        
        # Get performance metrics
        performance = await portfolio_manager.analyze_performance()
    
    # Display portfolio summary
    console.print("\n[bold cyan]Portfolio Summary[/bold cyan]")
    
    # Create a table for portfolio value
    value_table = Table(show_header=True, header_style="bold magenta")
    value_table.add_column("Metric", style="cyan")
    value_table.add_column("Value", style="green")
    
    value_table.add_row("Total Portfolio Value", f"${total_value:.2f}")
    value_table.add_row("Initial Portfolio Value", f"${performance.get('initial_value', 0):.2f}")
    
    total_return = performance.get('total_return', 0) * 100
    total_return_style = "green" if total_return >= 0 else "red"
    value_table.add_row("Total Return", f"[{total_return_style}]{total_return:.2f}%[/{total_return_style}]")
    
    annualized_return = performance.get('annualized_return', 0) * 100
    annualized_return_style = "green" if annualized_return >= 0 else "red"
    value_table.add_row("Annualized Return", f"[{annualized_return_style}]{annualized_return:.2f}%[/{annualized_return_style}]")
    
    value_table.add_row("Volatility", f"{performance.get('volatility', 0) * 100:.2f}%")
    value_table.add_row("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
    
    console.print(value_table)
    
    # Create a table for asset allocation
    console.print("\n[bold cyan]Asset Allocation[/bold cyan]")
    
    allocation_table = Table(show_header=True, header_style="bold magenta")
    allocation_table.add_column("Asset", style="cyan")
    allocation_table.add_column("Current", style="green")
    allocation_table.add_column("Target", style="blue")
    allocation_table.add_column("Deviation", style="yellow")
    allocation_table.add_column("Wallet Balance", style="green")
    allocation_table.add_column("Intents Balance", style="green")
    
    for token in portfolio_manager.target_allocation:
        current = current_allocation.get(token, 0)
        target = portfolio_manager.target_allocation[token]
        deviation: Decimal = Decimal(current) - Decimal(target)
        deviation_style = "green" if abs(deviation) <= Decimal(portfolio_manager.rebalance_threshold) else "red"
        
        allocation_table.add_row(
            token,
            f"{current * 100:.2f}%",
            f"{target * 100:.2f}%",
            f"[{deviation_style}]{deviation * 100:+.2f}%[/{deviation_style}]",
            f"{Decimal(wallet_balance.get(token, 0)):.6f}",
            f"{Decimal(intents_balance.get(token, 0)):.6f}"
        )
    
    console.print(allocation_table)
    
    # Show rebalance status
    needs_rebalance = portfolio_manager.needs_rebalancing(current_allocation)
    rebalance_status = "[bold red]Needs Rebalancing[/bold red]" if needs_rebalance else "[bold green]Balanced[/bold green]"
    console.print(f"\nPortfolio Status: {rebalance_status}")

async def run_rebalancer(portfolio_manager):
    """Run the portfolio rebalancer manually"""
    console.print(Panel("[bold blue]Portfolio Rebalancer[/bold blue]", expand=False))
    
    # Get current allocation
    with console.status("[bold green]Calculating current allocation...[/bold green]"):
        current_allocation = await portfolio_manager.calculate_current_allocation()
    
    # Check if rebalancing is needed
    needs_rebalance = portfolio_manager.needs_rebalancing(current_allocation)
    
    if not needs_rebalance:
        console.print("[bold green]Portfolio is already balanced. No rebalancing needed.[/bold green]")
        return
    
    # Show current vs target allocation
    console.print("\n[bold cyan]Current vs Target Allocation[/bold cyan]")
    
    allocation_table = Table(show_header=True, header_style="bold magenta")
    allocation_table.add_column("Asset", style="cyan")
    allocation_table.add_column("Current", style="green")
    allocation_table.add_column("Target", style="blue")
    allocation_table.add_column("Deviation", style="yellow")
    
    for token in portfolio_manager.target_allocation:
        current = Decimal(current_allocation.get(token, 0))
        target = Decimal(portfolio_manager.target_allocation[token])
        deviation = current - target
        deviation_style = "green" if abs(deviation) <= portfolio_manager.rebalance_threshold else "red"
        
        allocation_table.add_row(
            token,
            f"{current * 100:.2f}%",
            f"{target * 100:.2f}%",
            f"[{deviation_style}]{deviation * 100:+.2f}%[/{deviation_style}]"
        )
    
    console.print(allocation_table)
    
    # Confirm rebalance
    confirm = Confirm.ask("\nDo you want to proceed with rebalancing?")
    if not confirm:
        console.print("[bold yellow]Rebalancing cancelled.[/bold yellow]")
        return
    
    # Execute rebalance
    with console.status("[bold green]Executing portfolio rebalance...[/bold green]"):
        success = await portfolio_manager.manage_portfolio()
    
    if success:
        console.print("[bold green]Portfolio rebalanced successfully![/bold green]")
    else:
        console.print("[bold red]Portfolio rebalance failed![/bold red]")


async def deposit_to_intents(portfolio_manager, tokens_data):
    """Deposit tokens to intents"""
    console.print(Panel("[bold blue]Deposit to Intents[/bold blue]", expand=False))
    
    # Display available tokens
    console.print("[bold cyan]Available Tokens[/bold cyan]")
    
    tokens_table = Table(show_header=True, header_style="bold magenta")
    tokens_table.add_column("Symbol", style="cyan")
    tokens_table.add_column("Blockchain", style="green")
    tokens_table.add_column("Balance", style="yellow")
    
    valid_symbols = []
    for token in tokens_data:
        symbol = token["symbol"]
        wallet_balance = Decimal(portfolio_manager.wallet_balance.get(symbol, 0))
        if symbol not in valid_symbols:
            valid_symbols.append(symbol)
            tokens_table.add_row(symbol, token["blockchain"], f"{wallet_balance:.6f}")
    
    console.print(tokens_table)
    
    # Ask for token symbol
    token_symbol = Prompt.ask(
        "Enter token symbol to deposit",
        choices=valid_symbols,
        case_sensitive=False
    )
    
    # Ask for amount
    amount = Prompt.ask(
        f"Enter amount of {token_symbol} to deposit",
        default = str(Decimal(portfolio_manager.wallet_balance.get(token_symbol, 0))), 
        case_sensitive=False
    )
    
    try:
        amount = Decimal(amount)
    except ValueError:
        console.print("[bold red]Invalid amount. Please enter a numerical value.[/bold red]")
        return
    
    if amount <= 0:
        console.print("[bold red]Amount must be greater than 0.[/bold red]")
        return
    
    # Confirm the deposit
    confirm = Confirm.ask(f"Deposit {amount} {token_symbol} to intents?")
    if not confirm:
        console.print("[bold yellow]Deposit cancelled.[/bold yellow]")
        return
    
    # Execute the deposit
    with console.status(f"[bold green]Depositing {amount} {token_symbol} to intents. This can take upto 15 mins...[/bold green]"):
        if token_symbol.upper() == "ZEC":
            success = await intents._deposit_to_intents(tokens_data, amount, portfolio_manager.zec_account_id,token_symbol)
        else:
            success = await intents._deposit_to_intents(tokens_data, amount, portfolio_manager.account_id,token_symbol)
            
    if success:
        console.print(f"[bold green]Successfully deposited {amount} {token_symbol} to intents![/bold green]")
        
    else:
        console.print(f"[bold red]Failed to deposit {amount} {token_symbol} to intents![/bold red]")

async def withdraw_from_intents(portfolio_manager: PortfolioManager, tokens_data):
    """Withdraw tokens from intents"""
    console.print(Panel("[bold blue]Withdraw from Intents[/bold blue]", expand=False))
    
    # Display available tokens in intents
    console.print("[bold cyan]Intents Balance[/bold cyan]")
    
    intents_table = Table(show_header=True, header_style="bold magenta")
    intents_table.add_column("Symbol", style="cyan")
    intents_table.add_column("Balance", style="green")
    
    valid_symbols = []
    for token, balance in portfolio_manager.intents_balance.items():
        balance = Decimal(balance)
        if (balance) > 0:
            valid_symbols.append(token)
            intents_table.add_row(token, f"{balance:.6f}")
    
    console.print(intents_table)
    
    if not valid_symbols:
        console.print("[bold yellow]No tokens available to withdraw from intents.[/bold yellow]")
        return
    
    # Ask for token symbol
    token_symbol = Prompt.ask(
        "Enter token symbol to withdraw",
        choices=valid_symbols,
        case_sensitive=False
    )
    
    # Ask for amount
    max_amount = Decimal(portfolio_manager.intents_balance.get(token_symbol, 0))
    amount = Prompt.ask(
        f"Enter amount of {token_symbol} to withdraw (max: {max_amount:.6f})",
        default=str(max_amount)
    )
    
    try:
        amount = Decimal(amount)
    except ValueError:
        console.print("[bold red]Invalid amount. Please enter a numerical value.[/bold red]")
        return
    
    if amount <= 0:
        console.print("[bold red]Amount must be greater than 0.[/bold red]")
        return
    
    if amount > max_amount:
        console.print(f"[bold red]Amount exceeds available balance of {max_amount:.6f} {token_symbol}.[/bold red]")
        return
    
    min_amount = 0
    for tk in tokens_data:
        if tk["symbol"] == token_symbol:
            decimals = int(tk["decimals"])
            min_amount = Decimal(tk["min_withdraw_amount"]) / Decimal(10 ** decimals)
            
    if amount < min_amount:
        console.print(f"[bold red]Amount to withdraw is less than the minimum amount possible to withdraw {min_amount} {token_symbol} [/bold red]")
        return
    
    # Ask for receiver ID
    
    
    if (token_symbol.upper() == "ZEC"):
        zec_default = (portfolio_manager.zec_account_id)[:6] +  "..." + (portfolio_manager.zec_account_id)[-4:]
        receiver_id = Prompt.ask("Enter receiverId", default=zec_default)
        if (receiver_id == zec_default):
            receiver_id = portfolio_manager.zec_account_id
    else:
        receiver_id = Prompt.ask("Enter receiverId", default=portfolio_manager.account_id)
    
    # Confirm the withdrawal
    confirm = Confirm.ask(f"Withdraw {amount} {token_symbol} to {receiver_id}?")
    if not confirm:
        console.print("[bold yellow]Withdrawal cancelled.[/bold yellow]")
        return
    
    # Execute the withdrawal
    with console.status(f"[bold green]Withdrawing {amount} {token_symbol} to {receiver_id}. This can take upto 15 mins ...[/bold green]"):
        if token_symbol.upper() == "ZEC":
            success = await zcash.withdraw(token_symbol, amount, receiver_id, tokens_data)
        else:
            success = await intents.withdraw_from_intents(token_symbol, amount, receiver_id, tokens_data)
    
    if success:
        console.print(f"[bold green]Successfully withdrew {amount} {token_symbol} to {receiver_id}![/bold green]")
        
    else:
        console.print(f"[bold red]Failed to withdraw {amount} {token_symbol}![/bold red]")

async def rebalance_task(portfolio_manager):
    """Background task to run the rebalancer every 2 minutes"""
    while True:
        try:
            # Calculate current allocation
            current_allocation = await portfolio_manager.calculate_current_allocation()
            
            # Check if rebalancing is needed
            if portfolio_manager.needs_rebalancing(current_allocation):
                console.print("\n[bold yellow]Auto-rebalance triggered![/bold yellow]")
                
                # Execute rebalance
                success = await portfolio_manager.execute_rebalance()
                
                if success:
                    console.print("[bold green]Auto-rebalance completed successfully![/bold green]")
                else:
                    console.print("[bold red]Auto-rebalance failed![/bold red]")
            
            # Wait for 2 minutes
            await asyncio.sleep(120)
        
        except Exception as e:
            console.print(f"[bold red]Error in rebalance task: {str(e)}[/bold red]")
            await asyncio.sleep(10)  # Wait a bit before retrying


async def display_status(portfolio_manager):
    """Display current portfolio status"""
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get portfolio state
    wallet_balance, intents_balance, total_value = await portfolio_manager.get_portfolio_state()
    
    # Get current allocation
    current_allocation = await portfolio_manager.calculate_current_allocation()
    
    # Check if rebalancing is needed
    needs_rebalance = portfolio_manager.needs_rebalancing(current_allocation)
    rebalance_status = "[bold red]Needs Rebalancing[/bold red]" if needs_rebalance else "[bold green]Balanced[/bold green]"
    
    console.print(f"\n[bold cyan]Status as of {current_time}[/bold cyan]")
    console.print(f"Total Portfolio Value: ${total_value:.2f}")
    console.print(f"Portfolio Status: {rebalance_status}")
    
async def view_balances(portfolio_manager):
    """View wallet and intents balances"""
    console.print(Panel("[bold blue]View Balances[/bold blue]", expand=False))
    
    wallet_balance = portfolio_manager.wallet_balance
    intents_balance = portfolio_manager.intents_balance
    
    balance_table = Table(show_header=True, header_style="bold magenta")
    balance_table.add_column("Token", style="cyan")
    balance_table.add_column("Wallet Balance", style="green")
    balance_table.add_column("Intents Balance", style="yellow")
    balance_table.add_column("Total Balance", style="blue")
    
    for token in set(list(wallet_balance.keys()) + list(intents_balance.keys())):
        wallet_amount = Decimal(wallet_balance.get(token, 0))
        intents_amount = Decimal(intents_balance.get(token, 0))
        total_amount = wallet_amount + intents_amount
        balance_table.add_row(
            token,
            f"{wallet_amount:.6f}",
            f"{intents_amount:.6f}",
            f"{total_amount:.6f}"
        )
    
    console.print(balance_table)

async def make_intents_swap(portfolio_manager, tokens_data):
    """Make a swap using intents"""
    console.print(Panel("[bold blue]Make Intents Swap[/bold blue]", expand=False))
      
    # Display available tokens in intents
    console.print("[bold cyan]Intents Balance[/bold cyan]")
    
    intents_table = Table(show_header=True, header_style="bold magenta")
    intents_table.add_column("Symbol", style="cyan")
    intents_table.add_column("Balance", style="green")
    
    valid_symbols = []
    for token, balance in portfolio_manager.intents_balance.items():
        balance = Decimal(balance)
        if (balance) > 0:
            valid_symbols.append(token)
            intents_table.add_row(token, f"{balance:.6f}")
    
    console.print(intents_table)
    
    if not valid_symbols:
        console.print("[bold yellow]No tokens available to withdraw from intents.[/bold yellow]")
        return
    
    # Get user input
    token_in = Prompt.ask("Enter token symbol to swap from", choices=valid_symbols, case_sensitive=False)
    token_out = Prompt.ask("Enter token symbol to swap to", choices=valid_symbols, case_sensitive=False)
    
    max_amount = Decimal(portfolio_manager.intents_balance.get(token_in, 0))
    amount_in = Prompt.ask(f"Enter amount of {token_in} to swap", default=str(max_amount))
    
    try:
        amount_in = Decimal(amount_in)
    except ValueError:
        console.print("[bold red]Invalid amount. Please enter a numerical value.[/bold red]")
        return
    
    # Confirm the swap
    confirm = Confirm.ask(f"Swap {amount_in} {token_in} for {token_out}?")
    if not confirm:
        console.print("[bold yellow]Swap cancelled.[/bold yellow]")
        return
    
    # Execute the swap
    with console.status(f"[bold green]Executing swap: {amount_in} {token_in} to {token_out}...[/bold green]"):
        result = await intents.intent_swap(token_in, token_out, amount_in, tokens_data)
    
    if result:
        console.print(f"[bold green]Successfully swapped {amount_in} {token_in} for {result} {token_out}![/bold green]")
    else:
        console.print("[bold red]Swap failed![/bold red]")

async def make_complete_swap(portfolio_manager: PortfolioManager, tokens_data):
    """Make a complete swap (deposit to intents if needed, then swap)"""
    console.print(Panel("[bold blue]Make Complete Swap[/bold blue]", expand=False))
    
    # Display available tokens
    console.print("[bold cyan]Available Tokens[/bold cyan]")
    tokens_table = Table(show_header=True, header_style="bold magenta")
    tokens_table.add_column("Symbol", style="cyan")
    tokens_table.add_column("Blockchain", style="yellow")
    tokens_table.add_column("Wallet Balance", style="green")
    
    valid_symbols = []
    for token in tokens_data:
        symbol = token["symbol"]
        wallet_balance = Decimal(portfolio_manager.wallet_balance.get(symbol, 0))
        if wallet_balance > 0 and symbol not in valid_symbols:
            valid_symbols.append(symbol)
            tokens_table.add_row(symbol, token["blockchain"], f"{wallet_balance:.6f}")
    
    console.print(tokens_table)
    
    # Get user input
    token_in = Prompt.ask("Enter token symbol to swap from", choices=valid_symbols,case_sensitive=False)
    max_amount = Decimal(portfolio_manager.wallet_balance.get(token_in, 0))
    token_out = Prompt.ask("Enter token symbol to swap to", choices=valid_symbols,case_sensitive=False)
    amount_in = Prompt.ask(f"Enter amount of {token_in} to swap", default=str(max_amount))
    
    if (token_out.upper() == "ZEC"):
        zec_default = (portfolio_manager.zec_account_id)[:6] +  "..." + (portfolio_manager.zec_account_id)[-4:]
        receiver_id = Prompt.ask("Enter receiverId", default=zec_default)
        if (receiver_id == zec_default):
            receiver_id = portfolio_manager.zec_account_id
    else:
        receiver_id = Prompt.ask("Enter receiverId", default=portfolio_manager.account_id)
        
    try:
        amount_in = Decimal(amount_in)
    except ValueError:
        console.print("[bold red]Invalid amount. Please enter a numerical value.[/bold red]")
        return
    
    # Check if we need to deposit to intents first
    wallet_balance = Decimal(portfolio_manager.wallet_balance.get(token_in, 0))
    if wallet_balance < amount_in:
        console.print(f"[bold yellow]Insufficient balance in wallet. Available: {wallet_balance} {token_in}[/bold yellow]")
        return
    
    # Confirm the swap
    confirm = Confirm.ask(f"Swap {amount_in} {token_in} for {token_out}?")
    if not confirm:
        console.print("[bold yellow]Swap cancelled.[/bold yellow]")
        return
    
    # Execute the deposit if needed
    if wallet_balance >= amount_in:
        with console.status(f"[bold green]Depositing {amount_in} {token_in} to intents. This can take upto 15 mins ...[/bold green]"):
            if token_in.upper() == "ZEC":
                deposit_success = await intents._deposit_to_intents(tokens_data, amount_in, portfolio_manager.zec_account_id, token_in)
            else:
                deposit_success = await intents._deposit_to_intents(tokens_data, amount_in, portfolio_manager.account_id, token_in)
        
        if not deposit_success:
            console.print("[bold red]Deposit to intents failed![/bold red]")
            return
    
    # Execute the swap
    with console.status(f"[bold green]Executing swap: {amount_in} {token_in} to {token_out}...[/bold green]"):
        result = await intents.intent_swap(token_in, token_out, amount_in, tokens_data)
    
    if result:
        
        min_amount = 0
        for tk in tokens_data:
            if tk["symbol"] == token_out:
                decimals = int(tk["decimals"])
                min_amount = Decimal(tk["min_withdraw_amount"]) / Decimal(10 ** decimals)
                
        if result < min_amount:
            console.print(f"[bold red]Amount to withdraw is less than the minimum amount possible to withdraw {min_amount} {token_out} [/bold red]")
            return
            
        with console.status(f"[bold green]Executing Withdraw. This can take upto 15 mins...: {result} {token_out}...[/bold green]"):
            if token_out.upper() == "ZEC":
                result2 = await zcash.withdraw(token_out, result, receiver_id, tokens_data)
            else:
                result2 = await intents.withdraw_from_intents(token_out, result, receiver_id, tokens_data)
                
    if result and result2:
        console.print(f"[bold green]Successfully swapped {amount_in} {token_in} for {result} {token_out}![/bold green]")
    else:
        console.print("[bold red]Complete Swap failed![/bold red]")

async def main_menu():
    """Display the main menu and handle user input"""
    # Clear the screen
    console.clear()
    
    # Display welcome message
    console.print(Panel(
        "[bold blue]Z-Portfolio Manager[/bold blue]\n"
        "[italic]A powerful tool to manage your crypto portfolio[/italic]",
        expand=False,
        border_style="cyan"
    ))
    
    # Initialize portfolio manager
    portfolio_manager = await initialize_portfolio_manager()
    if not portfolio_manager:
        console.print("[bold red]Failed to initialize portfolio manager. Exiting...[/bold red]")
        return
    
    # Load tokens data for deposit/withdraw operations
    tokens_data = portfolio_manager.token_data
    if not tokens_data:
        console.print("[bold red]Failed to load tokens data. Some features may not work.[/bold red]")
    
    # Start background rebalance task
    console.print("[bold green]Starting auto-rebalance task (runs every 2 minutes)...[/bold green]")
    # rebalance_task_obj = asyncio.create_task(rebalance_task(portfolio_manager))
    
    try:
        while True:
            # Clear the screen
            console.clear()
            
            # Display header
            console.print(Panel(
                "[bold blue]Z-Portfolio Manager[/bold blue]\n"
                "[italic]A powerful tool to manage your crypto portfolio[/italic]",
                expand=False,
                border_style="cyan"
            ))
            
            # Display current status
            with console.status("[bold green]Checking Portfolio Status...[/bold green]"):
                portfolio_manager.token_data = await update_token_prices(portfolio_manager.token_data)
                await display_status(portfolio_manager)
            
            # Display menu options
            console.print("\n[bold green]Menu Options:[/bold green]")
            console.print("1. [cyan]View Balances[/cyan]")
            console.print("2. [cyan]View Portfolio Performance[/cyan]")
            console.print("3. [cyan]Make Complete Swap[/cyan]")
            console.print("4. [cyan]Deposit to Intents[/cyan]")
            console.print("5. [cyan]Make Intents Swap[/cyan]")
            console.print("6. [cyan]Withdraw from Intents[/cyan]")
            console.print("7. [cyan]Run Rebalancer[/cyan]")
            console.print("8. [red]Exit[/red]")
            
            # Get user choice
            choice = Prompt.ask("\n[bold]Choose an option[/bold]", choices=["1", "2", "3", "4", "5", "6", "7", "8"], default="1")
            
            # Clear the screen
            console.clear()
            
            if choice == "1":
                await view_balances(portfolio_manager)
            elif choice == "2":
                await view_performance(portfolio_manager)
            elif choice == "3":
                await make_complete_swap(portfolio_manager, tokens_data)
            elif choice == "4":
                await deposit_to_intents(portfolio_manager, tokens_data)
            elif choice == "5":
                await make_intents_swap(portfolio_manager, tokens_data)
            elif choice == "6":
                await withdraw_from_intents(portfolio_manager, tokens_data)
            elif choice == "7":
                await run_rebalancer(portfolio_manager)
            elif choice == "8":
                break
            
            # Wait for user to press enter before returning to menu
            Prompt.ask("\n[bold]Press Enter to return to the menu[/bold]", default="")
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting...[/bold yellow]")
    
    finally:
        # Cancel background task
        # console.print("[bold yellow]Stopping auto-rebalance task...[/bold yellow]")
        # rebalance_task_obj.cancel()
        # try:
        #     await rebalance_task_obj
        # except asyncio.CancelledError:
        #     pass
        
        console.print("[bold green]Thank you for using Z-Portfolio Manager![/bold green]")


if __name__ == "__main__":
    try:
        asyncio.run(main_menu())
    
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
