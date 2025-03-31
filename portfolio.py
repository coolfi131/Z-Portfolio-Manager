import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import utils
import intents
from decimal import Decimal

class PortfolioManager:
    """
    Portfolio Manager for NEAR Intents.
    Manages a diversified portfolio of crypto assets using NEAR Intents.
    Implements various portfolio management strategies including rebalancing,
    diversification, and risk management.
    """
    
    def __init__(self, 
                 account_id: str, 
                 zec_account_id: str, 
                 target_allocation: Dict[str, float], 
                 rebalance_threshold: float = 0.03, 
                 rebalance_interval: Optional[timedelta] = None,
                 stop_loss: Optional[Dict[str, float]] = None,
                 take_profit: Optional[Dict[str, float]] = None,
                 token_data: Optional[Any] = None):
        """
        Initialize the Portfolio Manager.
        
        Args:
            account_id: The NEAR account ID to manage.
            target_allocation: A dictionary mapping token symbols to their target 
                               allocation percentage (0-1).
            rebalance_threshold: The threshold (as a percentage, e.g., 0.03 for 3%) 
                                at which to trigger a rebalance.
            rebalance_interval: The interval at which to trigger a rebalance.
                                If None, only threshold-based rebalancing is used.
            stop_loss: A dictionary mapping token symbols to their stop-loss 
                      percentage (0-1).
            take_profit: A dictionary mapping token symbols to their take-profit 
                         percentage (0-1).
            token_data: Token metadata and pricing information.
        """
        self.account_id = account_id
        self.zec_account_id = zec_account_id
        self.target_allocation = target_allocation
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_interval = rebalance_interval
        self.stop_loss = stop_loss or {}
        self.take_profit = take_profit or {}
        self.token_data = token_data
        
        # Validate target allocation
        if abs(sum(target_allocation.values()) - 1.0) > 0.0001:
            raise ValueError("Target allocation must sum to 1.0")
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Set the logging level
        file_handler = logging.FileHandler('output.log')  # Log to 'output.log'
        file_handler.setLevel(logging.INFO)  # Set level for file logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info("Portfolio Manager initialized")
        
        # Initialize portfolio state
        self.wallet_balance = {}
        self.intents_balance = {}
        self.last_rebalance_time = datetime.now()
        
        self.current_portfolio_value = 0
        self.performance_history = []
        
        # Initialize performance metrics
        self.initial_portfolio_value = 0
        try:
            with open("metadata.json", "r") as f:
                metadata = json.load(f)
                if (Decimal(metadata["initial_portfolio_value"]) > 0):
                    self.initial_portfolio_value = Decimal(metadata["initial_portfolio_value"])
                    
        except Exception as e:
            self.logger.error(f"metadata.json cannot be opened: {str(e)}")
            return None
        
        
    async def get_portfolio_state(self) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Get the current state of the portfolio, including wallet balance,
        intents balance, and total portfolio value.
        
        Returns:
            A tuple containing the wallet balance, intents balance, and total portfolio value.
        """
        try:
            
            # Get wallet balance
            wallet_balance_result = await utils._wallet_balance(self.account_id, self.token_data)
            wallet_balance_result = json.loads(wallet_balance_result)
            self.wallet_balance = {item['symbol']: item['balance'] for item in wallet_balance_result}
            
            # Get intents balance
            intents_balance_result = await utils._Intents_balance(self.account_id, self.token_data)
            self.intents_balance = {item['TOKEN']: item['AMOUNT'] for item in intents_balance_result}
            
            # Calculate total portfolio value
            total_value = 0
            for token, balance in self.wallet_balance.items():
                match = [obj for obj in self.token_data if obj["symbol"] == token.upper()]
                if not match:
                    continue
                token_dat = match[0]
                price = Decimal(Decimal(token_dat['price']))
                balance = Decimal(balance)
                total_value += balance * price
            
            for token, balance in self.intents_balance.items():
                match = [obj for obj in self.token_data if obj["symbol"] == token.upper()]
                if not match:
                    continue
                token_dat = match[0]
                price = Decimal(Decimal(token_dat['price']))
                balance = Decimal(balance)
                total_value += balance * price
        
            # Update portfolio value
            self.current_portfolio_value = total_value
            if self.initial_portfolio_value == 0:
                self.initial_portfolio_value = total_value
                try:
                    metadata = {
                        "initial_portfolio_value": str(self.initial_portfolio_value),
                    }
                    metadata = json.dumps(metadata)
                    with open("metadata.json", "w") as f:
                        f.write(metadata)
                            
                except Exception as e:
                    self.logger.error(f"Error updating metadata: {str(e)}")
                    pass
            
            # Record portfolio state in history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': total_value,
                'wallet_balance': self.wallet_balance.copy(),
                'intents_balance': self.intents_balance.copy(),
                'token_data': self.token_data.copy(),
            })
            
            return self.wallet_balance, self.intents_balance, total_value
        except Exception as e:
            self.logger.error(f"Error getting portfolio state: {str(e)}")
            raise
    
    async def calculate_current_allocation(self) -> Dict[str, float]:
        """
        Calculate the current allocation of assets in the portfolio.
        
        Returns:
            A dictionary mapping token symbols to their current allocation percentage.
        """
        try:
            _, _, total_value = await self.get_portfolio_state()
            
            if total_value == 0:
                return {token: 0 for token in self.target_allocation}
            
            current_allocation = {}
            for token in self.target_allocation:
                wallet_balance = Decimal(self.wallet_balance.get(token, 0))
                intents_balance = Decimal(self.intents_balance.get(token, 0))
                total_balance = wallet_balance + intents_balance
                
                for token_dat in self.token_data:
                    token_sym = token_dat["symbol"]
                    if (token_sym == token):
                        price = Decimal(token_dat['price'])
                        token_value = total_balance * price
                        current_allocation[token] = token_value / total_value

            return current_allocation
        except Exception as e:
            self.logger.error(f"Error calculating current allocation: {str(e)}")
            raise

    def needs_rebalancing(self, current_allocation: Dict[str, float]) -> bool:
        """
        Determine if the portfolio needs to be rebalanced based on the current allocation
        and the rebalancing threshold.
        
        Args:
            current_allocation: A dictionary mapping token symbols to their current allocation percentage.
        
        Returns:
            True if the portfolio needs to be rebalanced, False otherwise.
        """
        # Check if time-based rebalancing is due
        if self.rebalance_interval is not None:
            time_since_last_rebalance = datetime.now() - self.last_rebalance_time
            if time_since_last_rebalance >= self.rebalance_interval:
                self.logger.info("Time-based rebalancing triggered")
                return True
        
        # Check if threshold-based rebalancing is needed
        for token, target in self.target_allocation.items():
            current = current_allocation.get(token, 0)
            deviation:Decimal = abs(Decimal(current) - Decimal(target))
            if deviation > Decimal(self.rebalance_threshold):
                self.logger.info(f"Threshold-based rebalancing triggered for {token}: "
                                f"current={current:.4f}, target={target:.4f}, deviation={deviation:.4f}")
                return True
        
        return False
    
    async def execute_rebalance(self) -> bool:
        """
        Execute a portfolio rebalance to align the current allocation with the target allocation.
        
        Returns:
            True if the rebalance was successful, False otherwise.
        """
        try:
            self.logger.info("Starting portfolio rebalance")
            
            # Get current state
            current_allocation = await self.calculate_current_allocation()
            if not self.needs_rebalancing(current_allocation):
                self.logger.info("Portfolio is already balanced, no rebalance needed")
                return True
            
            # Calculate required adjustments
            adjustments = {}
            for token, target in self.target_allocation.items():
                current = Decimal(current_allocation.get(token, 0))
                deviation = Decimal(target) - current
                if abs(deviation) > 0.0001:  # Small epsilon to avoid rounding errors
                    adjustments[token] = deviation
            
            # Sort adjustments by absolute deviation (largest first)
            sorted_adjustments = sorted(
                adjustments.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Execute the adjustments
            for token, deviation in sorted_adjustments:
                if (deviation < 0) and (token != "USDC"):
                    await self._sell_token(token, abs(deviation))
                    
            new_allocation = await self.calculate_current_allocation()
            
            for token, deviation in sorted_adjustments:
                if (deviation > 0) and (token != "USDC"):
                    await self._buy_token(token, deviation)
            
            # Update last rebalance time
            self.last_rebalance_time = datetime.now()
            
            # Verify the rebalance
            new_allocation = await self.calculate_current_allocation()
            for token, target in self.target_allocation.items():
                current = Decimal(new_allocation.get(token, 0))
                deviation = abs(current - Decimal(target))
                if deviation > self.rebalance_threshold:
                    self.logger.warning(f"Rebalance did not achieve target for {token}: "
                                     f"current={current:.4f}, target={target:.4f}, deviation={deviation:.4f}")
            
            self.logger.info("Portfolio rebalance completed")
            return True
        except Exception as e:
            self.logger.error(f"Error executing rebalance: {str(e)}")
            return False

    async def _buy_token(self, token: str, allocation_percentage: float) -> bool:
        """
        Buy a token to increase its allocation in the portfolio.
        
        Args:
            token: The token symbol to buy.
            allocation_percentage: The percentage of the portfolio value to allocate to this token.
        
        Returns:
            True if the buy was successful, False otherwise.
        """
        try:
            match = [obj for obj in self.token_data if obj["symbol"] == token.upper()]
            if not match:
                return False
            token_dat = match[0]
            
            total_value = Decimal(self.current_portfolio_value)
            source_token = "USDC"
            source_amount = total_value * Decimal(allocation_percentage)
            
            if source_token is None or source_amount <= 0:
                self.logger.warning(f"Could not find a suitable source token for buying {token}")
                return False
            
            remianing:Decimal = source_amount - Decimal(self.intents_balance.get(source_token, 0))
        
            # First deposit to intents if needed
            if (Decimal(self.wallet_balance.get(source_token, 0)) >= remianing) and (remianing > 0):
                # We have enough in the wallet
                
                sender = self.account_id
                if source_token == "ZEC":
                    sender = self.zec_account_id
                    
                await (intents._deposit_to_intents(
                    self.token_data, remianing, sender, source_token
                ))
            else:
                # We need to use what's in intents
                # Ensure the source amount doesn't exceed what we have in intents
                intents_balance = Decimal(self.intents_balance.get(source_token, 0))
                if intents_balance < source_amount:
                    source_amount = intents_balance
            
            # Execute the swap
            await (intents.intent_swap(
                source_token, token, source_amount, self.token_data
            ))
            
            self.logger.info(f"Bought {token} with {source_amount} {source_token}")
            print(f"Bought {token} with {source_amount} {source_token}")
            return True
        except Exception as e:
            self.logger.error(f"Error buying token {token}: {str(e)}")
            return False
    
    async def _sell_token(self, token: str, allocation_percentage: float) -> bool:
        """
        Sell a token to decrease its allocation in the portfolio.
        
        Args:
            token: The token symbol to sell.
            allocation_percentage: The percentage of the portfolio value to reduce this token by.
        
        Returns:
            True if the sell was successful, False otherwise.
        """
        try:
            # Calculate amount to sell
            # _, _, total_value = await self.get_portfolio_state()
            
            
            total_value = self.current_portfolio_value
            amount_to_sell_value = total_value * allocation_percentage
            
            # Determine which token to swap to
            # For simplicity, we'll use a stablecoin or the token with the lowest allocation
            # that is below its target
            target_token = "USDC"
            
            # Calculate how much of the token to sell
            for token_dat in self.token_data:
                token_sym = token_dat["symbol"]
                if (token_sym == token):
                    token_price = Decimal(token_dat['price'])
                    
            # token_price = self.token_data.get(token, {}).get('price', 0)
            if token_price <= 0:
                self.logger.warning(f"Invalid price for {token}")
                return False
            
            amount_to_sell = amount_to_sell_value / token_price
            remaining : Decimal = amount_to_sell - Decimal(self.intents_balance.get(token, 0))
            
            # First deposit to intents if needed
            if (Decimal(self.wallet_balance.get(token, 0)) >= remaining) and (remaining > 0):
                
                sender = self.account_id
                if token == "ZEC":
                    sender = self.zec_account_id
                    
                await (intents._deposit_to_intents(
                    self.token_data, remaining, sender, token
                ))
            else:
                # We need to use what's in intents
                # Ensure the amount doesn't exceed what we have in intents
                intents_balance = Decimal(self.intents_balance.get(token, 0))
                if intents_balance < amount_to_sell:
                    amount_to_sell = intents_balance
            
            # Execute the swap
            await (intents.intent_swap(
                token, target_token, amount_to_sell, self.token_data
            ))
            
            self.logger.info(f"Sold {amount_to_sell} {token} for {target_token}")
            print.info(f"Sold {amount_to_sell} {token} for {target_token}")
            return True
        except Exception as e:
            self.logger.error(f"Error selling token {token}: {str(e)}")
            return False
        
        
    async def check_stop_loss_take_profit(self) -> bool:
        """
        Check if any tokens have hit their stop-loss or take-profit targets
        and execute the corresponding actions.
        
        Returns:
            True if any actions were taken, False otherwise.
        """
        if not self.stop_loss and not self.take_profit:
            return False
        
        try:
            # Get portfolio history for price movement analysis
            if len(self.performance_history) < 2:
                return False
            
            # Get the latest and previous portfolio states
            latest = self.performance_history[-1]
            previous = self.performance_history[-2]
            
            actions_taken = False
            
            # Check each token for stop-loss/take-profit conditions
            for token in self.target_allocation:
                
                match = [obj for obj in self.token_data if obj["symbol"] == token.upper()]
                if not match:
                    continue
                token_dat = match[0]
                
                latest_price = Decimal(token_dat['price'])
                
                previous = self.performance_history[-2]['token_data']
                match = [obj for obj in previous if obj["symbol"] == token.upper()]
                previous_token_dat = match[0]
                
                previous_price = Decimal(previous_token_dat['price'])
                if previous_price <= 0:
                    continue
                
                # Calculate price change
                price_change = (latest_price - previous_price) / previous_price
                
                # Check stop-loss
                stop_loss_threshold = self.stop_loss.get(token)
                if stop_loss_threshold is not None and price_change <= -stop_loss_threshold:
                    self.logger.info(f"Stop-loss triggered for {token} (price change: {price_change:.4f})")
                    
                    # Sell the token (for simplicity, convert to a stablecoin)
                    target_token = None
                    for t in self.target_allocation:
                        if "USDC" in t:
                            target_token = t
                            break
                    
                    if target_token:
                        # Sell a portion of the token
                        total_balance = Decimal(Decimal(self.wallet_balance.get(token, 0)) + 
                                        Decimal(self.intents_balance.get(token, 0)))
                        sell_amount:Decimal = total_balance * 0.5  # Sell 50% to limit losses
                        
                        if sell_amount > 0:
                            # First deposit to intents if needed
                            if Decimal(self.wallet_balance.get(token, 0)) >= sell_amount:
                                sender = self.account_id
                                if token == "ZEC":
                                    sender = self.zec_account_id
                                    
                                await (intents._deposit_to_intents(
                                    self.token_data, sell_amount, sender, token
                                ))
                            else:
                                # Adjust sell amount to what's available in intents
                                intents_balance = Decimal(self.intents_balance.get(token, 0))
                                if intents_balance < sell_amount:
                                    sell_amount = intents_balance
                            
                            # Execute the swap
                            await (intents.intent_swap(
                                token, target_token, sell_amount, self.token_data
                            ))
                            
                            self.logger.info(f"Executed stop-loss: Sold {sell_amount} {token} for {target_token}")
                            actions_taken = True
                
                # Check take-profit
                take_profit_threshold = self.take_profit.get(token)
                if take_profit_threshold is not None and price_change >= take_profit_threshold:
                    self.logger.info(f"Take-profit triggered for {token} (price change: {price_change:.4f})")
                    
                    # Sell a portion of the token to lock in profits
                    target_token = None
                    for t in self.target_allocation:
                        if "USDC" in t:
                            target_token = t
                            break
                    
                    if target_token:
                        # Sell a portion of the token
                        total_balance = Decimal(Decimal(self.wallet_balance.get(token, 0)) + 
                                        Decimal(self.intents_balance.get(token, 0)))
                        sell_amount:Decimal = total_balance * 0.3  # Sell 30% to lock in profits
                        
                        if sell_amount > 0:
                            # First deposit to intents if needed
                            if Decimal(self.wallet_balance.get(token, 0)) >= sell_amount:
                                
                                sender = self.account_id
                                if token == "ZEC":
                                    sender = self.zec_account_id
                                    
                                await (intents._deposit_to_intents(
                                    self.token_data, sell_amount, sender, token
                                ))
                            else:
                                # Adjust sell amount to what's available in intents
                                intents_balance = Decimal(self.intents_balance.get(token, 0))
                                if intents_balance < sell_amount:
                                    sell_amount = intents_balance
                            
                            # Execute the swap
                            await (intents.intent_swap(
                                token, target_token, sell_amount, self.token_data
                            ))
                            
                            self.logger.info(f"Executed take-profit: Sold {sell_amount} {token} for {target_token}")
                            actions_taken = True
            
            return actions_taken
        except Exception as e:
            self.logger.error(f"Error checking stop-loss/take-profit: {str(e)}")
            return False

    async def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze the performance of the portfolio.
        
        Returns:
            A dictionary containing performance metrics.
        """
        try:
            if len(self.performance_history) < 2:
                return {
                    "total_return": 0,
                    "annualized_return": 0,
                    "volatility": 0,
                    "sharpe_ratio": 0,
                }
            
            # Calculate total return
            initial_value = Decimal(self.initial_portfolio_value)
            current_value = Decimal(self.current_portfolio_value)
            if initial_value <= 0:
                return {
                    "total_return": 0,
                    "annualized_return": 0,
                    "volatility": 0,
                    "sharpe_ratio": 0,
                }
            
            total_return = (current_value - initial_value) / initial_value
            
            # Calculate annualized return
            first_timestamp = self.performance_history[0]['timestamp']
            latest_timestamp = self.performance_history[-1]['timestamp']
            days = (latest_timestamp - first_timestamp).days
            if days < 1:
                days = 1
            annualized_return = (Decimal(1 + total_return) ** (Decimal(365 / days))) - 1
            
            # Calculate volatility
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i-1]['portfolio_value']
                curr_value = self.performance_history[i]['portfolio_value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if len(returns) > 0:
                volatility = Decimal(np.std(returns)) * Decimal(np.sqrt(252))  # Annualized volatility
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                if volatility > 0:
                    sharpe_ratio = annualized_return / volatility
                else:
                    sharpe_ratio = 0
            else:
                volatility = 0
                sharpe_ratio = 0
            
            return {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "current_value": current_value,
                "initial_value": initial_value,
                "days": days,
            }
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {
                "total_return": 0,
                "annualized_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "error": str(e),
            }

    async def run_maintenance(self) -> bool:
        """
        Run routine maintenance tasks for the portfolio.
        
        Returns:
            True if maintenance was successful, False otherwise.
        """
        try:
            self.logger.info("Running portfolio maintenance")
            
            # Check for stop-loss/take-profit conditions
            sl_tp_actions = await self.check_stop_loss_take_profit()
            
            # Check if rebalancing is needed
            await self.execute_rebalance()
            
            return True
        except Exception as e:
            self.logger.error(f"Error running maintenance: {str(e)}")
            return False
    

    async def check_market_trends(self) -> Dict[str, str]:
        """
        Analyze market trends to inform portfolio decisions.
        
        Returns:
            A dictionary mapping token symbols to their trend ('bullish', 'bearish', or 'neutral').
        """
        try:
            # For a production system, you would integrate with a price feed API
            # and implement proper technical analysis
            # This is a simplified placeholder
            trends = {}
            for token in self.target_allocation:
                match = [obj for obj in self.token_data if obj["symbol"] == token.upper()]
                if not match:
                    trends[token] = 'neutral'
                    continue
                
                token_dat = match[0]
                
                # Simple trend analysis based on price history
                # In a real system, you would use proper indicators like MACD, RSI, etc.
                if len(self.performance_history) >= 3:
                    latest = self.performance_history[-1]
                    previous = self.performance_history[-2]['token_data']
                    match = [obj for obj in previous if obj["symbol"] == token.upper()]
                    previous_token_dat = match[0]
                    
                    oldest = self.performance_history[-3]['token_data']
                    match = [obj for obj in oldest if obj["symbol"] == token.upper()]
                    oldest_token_dat = match[0]
                    
                    # Get token prices
                    latest_price = Decimal(token_dat['price'])
                    
                    
                    # Get historical prices (simplified)
                    previous_price = Decimal(previous_token_dat['price']) 
                    oldest_price = Decimal(oldest_token_dat['price'])
                    
                    # Simple trend analysis
                    if latest_price > previous_price and previous_price > oldest_price:
                        trends[token] = 'bullish'
                    elif latest_price < previous_price and previous_price < oldest_price:
                        trends[token] = 'bearish'
                    else:
                        trends[token] = 'neutral'
                else:
                    trends[token] = 'neutral'
            
            return trends
        except Exception as e:
            self.logger.error(f"Error checking market trends: {str(e)}")
            return {token: 'neutral' for token in self.target_allocation}
    
    async def adapt_allocation_to_trends(self) -> Dict[str, float]:
        """
        Adapt the target allocation based on market trends.
        
        Returns:
            The adjusted target allocation based on market trends.
        """
        try:
            trends = await self.check_market_trends()
            adjusted_allocation = self.target_allocation.copy()
            # Adjust allocations based on trends
            for token, trend in trends.items():
                if trend == 'bullish':
                    # Increase allocation for bullish tokens
                    adjusted_allocation[token] *= 1.1  # Increase by 10%
                elif trend == 'bearish':
                    # Decrease allocation for bearish tokens
                    adjusted_allocation[token] *= 0.9  # Decrease by 10%
            
            # Normalize allocations to ensure they sum to 1
            total = sum(adjusted_allocation.values())
            adjusted_allocation = {k: v / total for k, v in adjusted_allocation.items()}
            
            return adjusted_allocation
        
        except Exception as e:
            self.logger.error(f"Error adapting allocation to trends: {str(e)}")
            return self.target_allocation

    async def optimize_yield(self) -> None:
        """
        Optimize the portfolio for yield opportunities.
        """
        try:
            # In a real system, you would integrate with DeFi protocols to find yield opportunities
            # This is a simplified placeholder
            self.logger.info("Optimizing portfolio for yield")
            
            # Example: Move a portion of stablecoins to a yield-generating protocol
            stablecoins = [token for token in self.target_allocation if "USDC" in token]
            for stablecoin in stablecoins:
                balance = Decimal(self.wallet_balance.get(stablecoin, 0)) + Decimal(self.intents_balance.get(stablecoin, 0))
                if balance > 0:
                    yield_amount = Decimal(balance) * Decimal(0.2)  # Move 20% to yield-generating protocol
                    self.logger.info(f"Moving {yield_amount} {stablecoin} to yield-generating protocol")
                    # Here you would integrate with a specific DeFi protocol
        except Exception as e:
            self.logger.error(f"Error optimizing yield: {str(e)}")

    async def manage_portfolio(self) -> None:
        """
        Enhanced main method to manage the portfolio.
        This should be called periodically to ensure the portfolio is properly managed.
        """
        try:
            self.logger.info("Starting enhanced portfolio management cycle")
            
            # Check market trends and adapt allocation
            adjusted_allocation = await self.adapt_allocation_to_trends()
            self.target_allocation = adjusted_allocation
            
            # Run maintenance tasks
            await self.run_maintenance()
            
            # Optimize for yield
            await self.optimize_yield()
            
            # Analyze and log performance
            performance = await self.analyze_performance()
            self.logger.info(f"Portfolio performance: total_return={performance['total_return']:.4f}, "
                        f"annualized_return={performance['annualized_return']:.4f}, "
                        f"sharpe_ratio={performance['sharpe_ratio']:.4f}")
            
            self.logger.info("Enhanced portfolio management cycle completed")
            return True
        except Exception as e:
            self.logger.error(f"Error managing portfolio: {str(e)}")

