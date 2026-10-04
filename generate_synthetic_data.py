#!/usr/bin/env python3
"""
Synthetic Data Generation for Micro-Investment Platform (Optimized)

Generates realistic user profiles, micro-transaction histories, round-up amounts,
and wallet cash flows for training ML models.

Data Assumptions:
- User demographics: 18-70 years, diverse income levels, urban/suburban distribution
- Transaction frequency: 5-50 transactions/day per user
- Merchant categories: dining, transport, bills, shopping, entertainment, groceries, healthcare
- Round-up: nearest dollar, $0.01-$0.99 per transaction
- Wallet cash flows: deposits, withdrawals, round-up investments, fees
- Time range: 6 months of daily data
- Train/test split: 80/20 by user (no data leakage)
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta, date
import random
import os
import json

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

MERCHANT_CATEGORIES = {
    'dining': ['restaurant', 'cafe', 'fast_food', 'food_delivery', 'bar'],
    'transport': ['uber', 'lyft', 'public_transit', 'gas_station', 'parking', 'toll'],
    'bills': ['electricity', 'water', 'internet', 'phone', 'insurance', 'subscription'],
    'shopping': ['amazon', 'target', 'walmart', 'clothing', 'electronics', 'home_goods'],
    'entertainment': ['netflix', 'spotify', 'movie_theater', 'gaming', 'events', 'streaming'],
    'groceries': ['supermarket', 'wholesale_club', 'farmers_market', 'convenience_store'],
    'healthcare': ['pharmacy', 'doctor_visit', 'dental', 'vision', 'fitness', 'therapy']
}

CATEGORY_WEIGHTS = np.array([0.20, 0.15, 0.10, 0.15, 0.10, 0.20, 0.10])
CATEGORIES = list(MERCHANT_CATEGORIES.keys())

AMOUNT_RANGES = {
    'bills': (20, 300),
    'groceries': (15, 200),
    'dining': (5, 80),
    'transport': (3, 60),
    'shopping': (10, 500),
    'entertainment': (5, 50),
    'healthcare': (10, 200)
}

INCOME_BRACKETS = {
    'low': (20000, 40000),
    'lower_middle': (40000, 65000),
    'middle': (65000, 100000),
    'upper_middle': (100000, 150000),
    'high': (150000, 300000)
}
INCOME_WEIGHTS = np.array([0.15, 0.25, 0.30, 0.20, 0.10])
INCOME_BRACKET_KEYS = list(INCOME_BRACKETS.keys())

AGE_DISTRIBUTION = {
    'gen_z': (18, 26),
    'millennial': (27, 42),
    'gen_x': (43, 58),
    'boomer': (59, 70)
}
AGE_WEIGHTS = np.array([0.20, 0.35, 0.25, 0.20])
AGE_GROUP_KEYS = list(AGE_DISTRIBUTION.keys())


def generate_user_profiles(n_users=1000):
    """Generate synthetic user profiles with demographics and financial attributes."""
    age_groups = np.random.choice(AGE_GROUP_KEYS, size=n_users, p=AGE_WEIGHTS)
    ages = np.array([np.random.randint(*AGE_DISTRIBUTION[g]) for g in age_groups])
    
    income_brackets = np.random.choice(INCOME_BRACKET_KEYS, size=n_users, p=INCOME_WEIGHTS)
    annual_incomes = np.array([np.random.randint(*INCOME_BRACKETS[b]) for b in income_brackets])
    monthly_incomes = annual_incomes / 12
    disposable_incomes = monthly_incomes * np.random.uniform(0.2, 0.5, n_users)
    
    risk_tolerances = np.random.choice(['conservative', 'moderate', 'aggressive'], size=n_users, p=[0.3, 0.5, 0.2])
    investment_goals = np.random.choice(['emergency_fund', 'retirement', 'major_purchase', 'wealth_building'], 
                                         size=n_users, p=[0.25, 0.30, 0.20, 0.25])
    
    signup_dates = [fake.date_between(start_date='-2y', end_date='-30d') for _ in range(n_users)]
    kyc_statuses = np.random.choice(['verified', 'pending', 'rejected'], size=n_users, p=[0.85, 0.10, 0.05])
    account_types = np.random.choice(['individual', 'joint', 'custodial'], size=n_users, p=[0.80, 0.15, 0.05])
    
    return pd.DataFrame({
        'user_id': np.arange(1, n_users + 1),
        'age': ages,
        'age_group': age_groups,
        'annual_income': annual_incomes,
        'monthly_income': np.round(monthly_incomes, 2),
        'disposable_income': np.round(disposable_incomes, 2),
        'income_bracket': income_brackets,
        'risk_tolerance': risk_tolerances,
        'investment_goal': investment_goals,
        'signup_date': signup_dates,
        'kyc_status': kyc_statuses,
        'account_type': account_types,
        'is_active': np.array(kyc_statuses) == 'verified'
    })


def generate_transactions(users_df, days=180):
    """Generate daily micro-transaction histories for each user (vectorized)."""
    active_users = users_df[users_df['is_active']]['user_id'].values
    user_incomes = users_df.set_index('user_id').loc[active_users, 'monthly_income'].values
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start_date, end_date, freq='D').date
    
    all_txns = []
    txn_id = 0
    
    for idx, user_id in enumerate(active_users):
        daily_budget = user_incomes[idx] / 30 * np.random.uniform(0.3, 0.8)
        
        for current_date in date_range:
            n_txns = np.random.poisson(lam=12)
            n_txns = max(1, min(n_txns, 40))
            
            daily_spent = 0
            for _ in range(n_txns):
                if daily_spent > daily_budget * 1.5:
                    break
                
                category = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
                merchant = np.random.choice(MERCHANT_CATEGORIES[category])
                
                low, high = AMOUNT_RANGES[category]
                amount = round(np.random.uniform(low, high), 2)
                daily_spent += amount
                
                round_up = round(1 - (amount % 1), 2) if amount % 1 > 0 else 0
                if round_up == 1:
                    round_up = 0
                
                txn_id += 1
                all_txns.append({
                    'transaction_id': f"txn_{txn_id:010d}",
                    'user_id': user_id,
                    'date': current_date,
                    'merchant_category': category,
                    'merchant_name': merchant,
                    'amount': amount,
                    'round_up_amount': round_up,
                    'is_weekend': current_date.weekday() >= 5
                })
    
    return pd.DataFrame(all_txns)


def generate_wallet_cashflows(users_df, transactions_df):
    """Generate historical wallet cash flows (vectorized by user)."""
    active_users = users_df[users_df['is_active']]['user_id'].values
    user_incomes = users_df.set_index('user_id').loc[active_users, 'monthly_income'].values
    
    all_cashflows = []
    cf_id = 0
    
    for idx, user_id in enumerate(active_users):
        user_txns = transactions_df[transactions_df['user_id'] == user_id]
        if user_txns.empty:
            continue
        
        start_date = user_txns['date'].min()
        end_date = user_txns['date'].max()
        date_range = pd.date_range(start_date, end_date, freq='D').date
        
        wallet_balance = np.random.uniform(50, 500)
        invested_balance = 0
        monthly_income = user_incomes[idx]
        
        for current_date in date_range:
            daily_txns = user_txns[user_txns['date'] == current_date]
            
            if not daily_txns.empty:
                total_spent = daily_txns['amount'].sum()
                total_roundup = daily_txns['round_up_amount'].sum()
                
                wallet_balance -= total_spent
                invested_balance += total_roundup
                
                cf_id += 1
                all_cashflows.append({
                    'cashflow_id': f"cf_{cf_id:010d}",
                    'user_id': user_id,
                    'date': current_date,
                    'type': 'spend',
                    'amount': -round(total_spent, 2),
                    'wallet_balance': round(wallet_balance, 2),
                    'invested_balance': round(invested_balance, 2),
                    'round_up_invested': round(total_roundup, 2)
                })
                
                if total_roundup > 0:
                    cf_id += 1
                    all_cashflows.append({
                        'cashflow_id': f"cf_{cf_id:010d}",
                        'user_id': user_id,
                        'date': current_date,
                        'type': 'roundup_investment',
                        'amount': round(total_roundup, 2),
                        'wallet_balance': round(wallet_balance, 2),
                        'invested_balance': round(invested_balance, 2),
                        'round_up_invested': round(total_roundup, 2)
                    })
            
            if current_date.day == 1:
                deposit = monthly_income * np.random.uniform(0.1, 0.3)
                wallet_balance += deposit
                cf_id += 1
                all_cashflows.append({
                    'cashflow_id': f"cf_{cf_id:010d}",
                    'user_id': user_id,
                    'date': current_date,
                    'type': 'deposit',
                    'amount': round(deposit, 2),
                    'wallet_balance': round(wallet_balance, 2),
                    'invested_balance': round(invested_balance, 2),
                    'round_up_invested': 0
                })
            
            if current_date.day == 15 and np.random.random() < 0.3:
                withdrawal = wallet_balance * np.random.uniform(0.05, 0.2)
                wallet_balance -= withdrawal
                cf_id += 1
                all_cashflows.append({
                    'cashflow_id': f"cf_{cf_id:010d}",
                    'user_id': user_id,
                    'date': current_date,
                    'type': 'withdrawal',
                    'amount': -round(withdrawal, 2),
                    'wallet_balance': round(wallet_balance, 2),
                    'invested_balance': round(invested_balance, 2),
                    'round_up_invested': 0
                })
            
            if np.random.random() < 0.02:
                fee = np.random.uniform(1, 5)
                wallet_balance -= fee
                cf_id += 1
                all_cashflows.append({
                    'cashflow_id': f"cf_{cf_id:010d}",
                    'user_id': user_id,
                    'date': current_date,
                    'type': 'fee',
                    'amount': -round(fee, 2),
                    'wallet_balance': round(wallet_balance, 2),
                    'invested_balance': round(invested_balance, 2),
                    'round_up_invested': 0
                })
    
    return pd.DataFrame(all_cashflows)


def split_train_test(users_df, transactions_df, cashflows_df, test_ratio=0.2):
    """Split data by user_id to prevent data leakage."""
    user_ids = users_df[users_df['is_active']]['user_id'].unique()
    n_test = int(len(user_ids) * test_ratio)
    test_user_ids = set(np.random.choice(user_ids, n_test, replace=False))
    
    train_users = users_df[~users_df['user_id'].isin(test_user_ids)]
    test_users = users_df[users_df['user_id'].isin(test_user_ids)]
    
    train_txns = transactions_df[~transactions_df['user_id'].isin(test_user_ids)]
    test_txns = transactions_df[transactions_df['user_id'].isin(test_user_ids)]
    
    train_cf = cashflows_df[~cashflows_df['user_id'].isin(test_user_ids)]
    test_cf = cashflows_df[cashflows_df['user_id'].isin(test_user_ids)]
    
    return {
        'train': {'users': train_users, 'transactions': train_txns, 'cashflows': train_cf},
        'test': {'users': test_users, 'transactions': test_txns, 'cashflows': test_cf}
    }


def save_datasets(datasets, output_dir='data'):
    """Save train/test datasets to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        datasets[split]['users'].to_csv(os.path.join(split_dir, 'users.csv'), index=False)
        datasets[split]['transactions'].to_csv(os.path.join(split_dir, 'transactions.csv'), index=False)
        datasets[split]['cashflows'].to_csv(os.path.join(split_dir, 'cashflows.csv'), index=False)
        
        print(f"{split.capitalize()} set saved to {split_dir}/")
        print(f"  Users: {len(datasets[split]['users'])}")
        print(f"  Transactions: {len(datasets[split]['transactions'])}")
        print(f"  Cashflows: {len(datasets[split]['cashflows'])}")


def save_assumptions(output_dir='data'):
    """Save data assumptions and generation rules."""
    os.makedirs(output_dir, exist_ok=True)
    
    assumptions = {
        "generation_date": datetime.now().isoformat(),
        "n_users": 1000,
        "time_horizon_days": 180,
        "train_test_split": 0.8,
        "split_method": "by_user_id_no_leakage",
        "user_demographics": {
            "age_distribution": AGE_DISTRIBUTION,
            "age_weights": AGE_WEIGHTS.tolist(),
            "income_brackets": INCOME_BRACKETS,
            "income_weights": INCOME_WEIGHTS.tolist()
        },
        "transaction_patterns": {
            "daily_transaction_count": "Poisson(lambda=12), clipped to [1, 40]",
            "merchant_categories": MERCHANT_CATEGORIES,
            "category_weights": CATEGORY_WEIGHTS.tolist(),
            "amount_ranges_by_category": AMOUNT_RANGES,
            "round_up_rule": "nearest_dollar, 0.01-0.99 per transaction"
        },
        "wallet_cashflows": {
            "initial_balance_range": "[50, 500]",
            "monthly_deposit": "10-30% of monthly_income on day 1",
            "mid_month_withdrawal": "5-20% of balance, 30% probability on day 15",
            "fee_frequency": "2% daily probability, $1-5",
            "round_up_investment": "automatic on each transaction"
        },
        "user_attributes": {
            "risk_tolerance": ["conservative:30%, moderate:50%, aggressive:20%"],
            "investment_goal": ["emergency_fund:25%, retirement:30%, major_purchase:20%, wealth_building:25%"],
            "kyc_status": ["verified:85%, pending:10%, rejected:5%"],
            "account_type": ["individual:80%, joint:15%, custodial:5%"]
        }
    }
    
    with open(os.path.join(output_dir, 'data_assumptions.json'), 'w') as f:
        json.dump(assumptions, f, indent=2, default=str)
    
    print(f"Data assumptions saved to {output_dir}/data_assumptions.json")


def main():
    print("Generating synthetic data...")
    
    print("\n1. Generating user profiles...")
    users_df = generate_user_profiles(n_users=1000)
    print(f"   Generated {len(users_df)} users ({users_df['is_active'].sum()} active)")
    
    print("\n2. Generating transactions...")
    transactions_df = generate_transactions(users_df, days=180)
    print(f"   Generated {len(transactions_df)} transactions")
    
    print("\n3. Generating wallet cashflows...")
    cashflows_df = generate_wallet_cashflows(users_df, transactions_df)
    print(f"   Generated {len(cashflows_df)} cashflow records")
    
    print("\n4. Splitting train/test...")
    datasets = split_train_test(users_df, transactions_df, cashflows_df, test_ratio=0.2)
    
    print("\n5. Saving datasets...")
    save_datasets(datasets, output_dir='data')
    
    print("\n6. Saving assumptions...")
    save_assumptions(output_dir='data')
    
    print("\nDone!")


if __name__ == '__main__':
    main()