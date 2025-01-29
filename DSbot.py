import os
import logging
import sqlite3
import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, CallbackContext
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
import requests
from sklearn.ensemble import RandomForestClassifier
from tweepy import Client as TwitterClient

# Configuration
CONFIG = {
    "SOLANA_RPC": os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com"),
    "TELEGRAM_TOKEN": os.getenv("7718248122:AAFdOrGQdWVuHMPu9oNs7BbU1ItL8z6o5vM"),
    "TWITTER_BEARER": os.getenv("tSyPZ9XhZTe8Z9rY4HNgG4LIM:lHk8gQ8Dg346wBYWXPbdeQkQ8SEN4YNJzNuBNPKzGgWbVcWetQ"),
    "CHECK_INTERVAL": 5,  # minutes
    "RISK_AMOUNT": 1.0  # SOL amount for simulation
}

# Initialize database with proper schema
conn = sqlite3.connect('bot_data.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS tokens
             (address TEXT PRIMARY KEY, 
              name TEXT,
              symbol TEXT,
              created_at DATETIME,
              chain TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS historical_data
             (id INTEGER PRIMARY KEY,
              token_address TEXT,
              timestamp DATETIME,
              price REAL,
              volume REAL,
              liquidity REAL,
              social_mentions INTEGER,
              FOREIGN KEY(token_address) REFERENCES tokens(address))''')

c.execute('''CREATE TABLE IF NOT EXISTS users 
             (user_id INT PRIMARY KEY, 
              preferred_chain TEXT,
              risk_level INT)''')

conn.commit()

class DataCollector:
    def __init__(self):
        self.solana = Client(CONFIG["SOLANA_RPC"])
        self.twitter = TwitterClient(bearer_token=CONFIG["TWITTER_BEARER"])
        
    def fetch_historical_data(self, token_address, days=7):
        """Collect historical data for ML training"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        try:
            c.execute('''SELECT * FROM historical_data 
                       WHERE token_address=? AND timestamp BETWEEN ? AND ?''',
                    (token_address, start_time, end_time))
            return c.fetchall()
        except Exception as e:
            logging.error(f"Historical data fetch failed: {e}")
            return []

    def update_token_registry(self, token_address, chain="solana"):
        """Store token metadata in database"""
        try:
            # Check if token exists
            c.execute('SELECT 1 FROM tokens WHERE address=?', (token_address,))
            if not c.fetchone():
                # Fetch token metadata from Solana
                info = self.solana.get_account_info(token_address, commitment=Confirmed).value
                if info:
                    c.execute('''INSERT INTO tokens 
                               (address, created_at, chain)
                               VALUES (?, ?, ?)''',
                            (token_address, datetime.now(), chain))
                    conn.commit()
        except Exception as e:
            logging.error(f"Token registry update failed: {e}")

class RiskAnalyzer:
    def simulate_transaction(self, token_address):
        """Simulate buy/sell transaction to assess slippage"""
        try:
            # Get current price and liquidity
            dex_data = requests.get(
                f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            ).json()['pairs'][0]
            
            price = float(dex_data['priceUsd'])
            liquidity = float(dex_data['liquidity']['usd'])
            
            # Simulate buy
            buy_amount_usd = CONFIG["RISK_AMOUNT"] * price  # Convert SOL to USD
            slippage_buy = self.calculate_slippage(buy_amount_usd, liquidity)
            
            # Simulate sell
            sell_amount_usd = buy_amount_usd * (1 - slippage_buy/100)
            slippage_sell = self.calculate_slippage(sell_amount_usd, liquidity)
            
            # Calculate net result
            net_percent = ((1 - slippage_sell/100) / (1 + slippage_buy/100) - 1) * 100
            
            return {
                "slippage_buy": slippage_buy,
                "slippage_sell": slippage_sell,
                "net_percent": net_percent,
                "liquidity": liquidity
            }
        except Exception as e:
            logging.error(f"Transaction simulation failed: {e}")
            return None

    def calculate_slippage(self, amount_usd, liquidity):
        """Estimate slippage based on liquidity depth"""
        if liquidity == 0:
            return 100  # 100% slippage if no liquidity
        return min((amount_usd / liquidity) * 100, 50)  # Max 50% slippage

class MLTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()
        
    def prepare_training_data(self):
        """Prepare dataset from historical data"""
        try:
            c.execute('''SELECT h.*, t.created_at 
                       FROM historical_data h
                       JOIN tokens t ON h.token_address = t.address''')
            data = c.fetchall()
            
            df = pd.DataFrame(data, columns=[
                "id", "token_address", "timestamp", "price", 
                "volume", "liquidity", "mentions", "created_at"
            ])
            
            # Feature engineering
            df['price_change_1h'] = df.groupby('token_address')['price'].pct_change(periods=12)
            df['volume_change_1h'] = df.groupby('token_address')['volume'].pct_change(periods=12)
            df['mentions_1h'] = df.groupby('token_address')['mentions'].rolling(12).sum().values
            
            # Label: 1 if price increases 10% in next hour
            df['label'] = (df.groupby('token_address')['price'].shift(-12) / 
                          df['price'] > 1.1).astype(int)
            
            return df.dropna()
        except Exception as e:
            logging.error(f"Training data prep failed: {e}")
            return pd.DataFrame()

    def train_model(self):
        """Retrain ML model periodically"""
        df = self.prepare_training_data()
        if not df.empty:
            X = df[['price_change_1h', 'volume_change_1h', 'mentions_1h']]
            y = df['label']
            self.model.fit(X, y)
            logging.info("Model retrained successfully")

class EnhancedMemecoinTracker(MemecoinTracker):
    def __init__(self):
        super().__init__()
        self.data_collector = DataCollector()
        self.risk_analyzer = RiskAnalyzer()
        self.ml_trainer = MLTrainer()
        
        # Schedule ML retraining daily
        schedule.every().day.at("00:00").do(self.ml_trainer.train_model)

    def analyze_token(self, token_address):
        try:
            self.data_collector.update_token_registry(token_address)
            
            # Store historical data
            dex_data = requests.get(
                f"https://api.birdeye.so/public/token/{token_address}?chain=solana"
            ).json()['data']
            
            c.execute('''INSERT INTO historical_data
                       (token_address, timestamp, price, volume, liquidity, social_mentions)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (token_address, datetime.now(),
                     dex_data.get('price', 0),
                     dex_data.get('volume24h', {}).get('value', 0),
                     dex_data.get('liquidity', {}).get('value', 0),
                     self.get_social_sentiment(token_address)))
            conn.commit()
            
            # Risk assessment
            risk = self.risk_analyzer.simulate_transaction(token_address)
            if not risk or risk['net_percent'] < -5:
                return None
                
            # ML prediction
            features = self.get_ml_features(token_address)
            prediction = self.ml_trainer.model.predict([features])[0]
            
            return {
                "address": token_address,
                "prediction": prediction,
                "risk_assessment": risk,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logging.error(f"Enhanced analysis failed: {e}")
            return None

    def get_ml_features(self, token_address):
        """Get features from historical data"""
        data = self.data_collector.fetch_historical_data(token_address)
        if not data:
            return [0, 0, 0]
            
        df = pd.DataFrame(data, columns=[
            "id", "token_address", "timestamp", "price", 
            "volume", "liquidity", "mentions"
        ])
        
        return [
            df['price'].pct_change().mean(),
            df['volume'].pct_change().mean(),
            df['mentions'].sum()
        ]

class ImprovedTelegramBot(TelegramBot):
    def send_alert(self, signal):
        risk = signal.get('risk_assessment', {})
        message = (
            f"🚀 New Signal: {signal['address']}\n"
            f"✅ Prediction Confidence: {signal['prediction']*100:.1f}%\n"
            f"📉 Simulated Net: {risk.get('net_percent', 0):.1f}%\n"
            f"💧 Liquidity: ${risk.get('liquidity', 0):,.0f}\n"
            f"🔗 Trade: https://jup.ag/swap/SOL-{signal['address']}"
        )
        self.updater.bot.send_message(chat_id="@yourchannel", text=message)

    def check(self, update: Update, context: CallbackContext):
        token_address = context.args[0] if context.args else None
        if token_address:
            analysis = self.tracker.analyze_token(token_address)
            if analysis:
                reply = (f"🔍 Analysis for {token_address}:\n"
                        f"Simulated Profit: {analysis['risk_assessment']['net_percent']:.1f}%\n"
                        f"Liquidity: ${analysis['risk_assessment']['liquidity']:,.0f}")
            else:
                reply = "Token too risky or analysis failed"
            update.message.reply_text(reply)
        else:
            update.message.reply_text("Please provide token address")

if __name__ == "__main__":
    # Initialize everything
    conn = sqlite3.connect('bot_data.db')
    ml_trainer = MLTrainer()
    ml_trainer.train_model()  # Initial training
    
    bot = ImprovedTelegramBot()
    bot.run()