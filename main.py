"""
Main Pipeline: Micro-Investment Platform AI Engine

Orchestrates:
1. Auto-Savings & Micro-Roundup Engine
2. AI Credit Readiness Engine (XGBoost + SHAP)
3. Micro-Insurance Nudge Recommender
"""

import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Tuple
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.roundup import RoundupEngine, simulate_user_cashflow
from engine.credit_scoring import FeatureEngineer, CreditReadinessModel, train_credit_model
from engine.insurance_nudge import run_nudge_engine, NudgeEngine


def load_data(data_dir: str = 'data', split: str = 'train') -> Dict[str, pd.DataFrame]:
    """Load train or test datasets."""
    split_dir = os.path.join(data_dir, split)
    
    users = pd.read_csv(os.path.join(split_dir, 'users.csv'))
    transactions = pd.read_csv(os.path.join(split_dir, 'transactions.csv'))
    cashflows = pd.read_csv(os.path.join(split_dir, 'cashflows.csv'))
    
    for df in [users, transactions, cashflows]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        if 'signup_date' in df.columns:
            df['signup_date'] = pd.to_datetime(df['signup_date']).dt.date
    
    return {'users': users, 'transactions': transactions, 'cashflows': cashflows}


def run_roundup_engine(
    transactions_df: pd.DataFrame,
    cashflows_df: pd.DataFrame,
    roundup_base: int = 100
) -> pd.DataFrame:
    """Run roundup engine on transactions."""
    
    print("\n" + "="*60)
    print("ROUNDUP ENGINE: Processing transactions...")
    print("="*60)
    
    engine = RoundupEngine(roundup_base=roundup_base)
    processed = engine.process_transactions(transactions_df)
    
    summary = engine.get_savings_summary(processed)
    
    print(f"Processed {len(processed)} transactions for {summary['user_id'].nunique()} users")
    print(f"Total roundup savings: BDT {summary['total_roundup'].sum():,.2f}")
    print(f"Average savings rate: {summary['savings_rate_pct'].mean():.2f}%")
    print(f"Avg roundup per transaction: BDT {summary['avg_roundup_per_txn'].mean():.2f}")
    
    return processed


def run_credit_scoring(
    train_data: Dict,
    test_data: Dict,
    roundup_train: pd.DataFrame,
    roundup_test: pd.DataFrame,
    model_path: str = 'models/credit_model.joblib'
) -> Tuple[CreditReadinessModel, pd.DataFrame, pd.DataFrame]:
    """Train and evaluate credit readiness model."""
    
    print("\n" + "="*60)
    print("CREDIT READINESS ENGINE: Training model...")
    print("="*60)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model, train_metrics = train_credit_model(
        train_data['users'],
        train_data['transactions'],
        train_data['cashflows'],
        roundup_train,
        model_type='regression'
    )
    
    print("\nEvaluating on test set...")
    fe = FeatureEngineer()
    test_features = fe.create_user_features(
        test_data['users'],
        test_data['transactions'],
        test_data['cashflows'],
        roundup_test
    )
    
    X_test = model.prepare_features(test_features)
    y_test_reg, y_test_clf = model.create_targets(
        test_features, test_data['transactions'], test_data['cashflows']
    )
    
    test_preds = model.predict(X_test)
    test_rmse = np.sqrt(np.mean((test_preds - y_test_reg) ** 2))
    test_mae = np.mean(np.abs(test_preds - y_test_reg))
    
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    
    test_features['predicted_credit_score'] = test_preds
    test_features['actual_credit_score'] = y_test_reg
    test_features['prediction_error'] = test_preds - y_test_reg
    
    print("\nGenerating SHAP explanations...")
    shap_summary = model.get_global_shap_summary(X_test)
    print("\nGlobal SHAP Feature Importance (top 15):")
    print(shap_summary.head(15).to_string(index=False))
    
    sample_explanations = model.explain_batch(X_test[:5])
    print("\nSample Individual Explanations:")
    for exp in sample_explanations:
        print(f"\nUser {exp['user_index']}: Predicted Score = {exp['prediction']:.0f}")
        print(f"  Base Value: {exp['base_value']:.0f}")
        for feat in exp['top_features'][:5]:
            print(f"  {feat['feature']}: SHAP={feat['shap_value']:.1f} (value={feat['feature_value']:.1f}) [{feat['impact']}]")
    
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    return model, test_features, shap_summary


def run_insurance_nudges(
    test_data: Dict,
    test_features: pd.DataFrame,
    output_path: str = 'outputs/insurance_nudges.csv'
) -> pd.DataFrame:
    """Run insurance nudge recommender."""
    
    print("\n" + "="*60)
    print("MICRO-INSURANCE NUDGE ENGINE: Generating recommendations...")
    print("="*60)
    
    nudges_df = run_nudge_engine(
        test_data['users'],
        test_features,
        output_path=output_path
    )
    
    return nudges_df


def generate_user_report(
    user_id: int,
    data: Dict,
    roundup_df: pd.DataFrame,
    credit_model: CreditReadinessModel,
    features_df: pd.DataFrame
) -> Dict:
    """Generate comprehensive report for a single user."""
    
    user_info = data['users'][data['users']['user_id'] == user_id].iloc[0]
    user_txns = data['transactions'][data['transactions']['user_id'] == user_id]
    user_cf = data['cashflows'][data['cashflows']['user_id'] == user_id]
    user_ru = roundup_df[roundup_df['user_id'] == user_id]
    user_feat = features_df[features_df['user_id'] == user_id]
    
    if user_feat.empty:
        return {'error': 'User not found in features'}
    
    roundup_engine = RoundupEngine()
    cashflow_sim = simulate_user_cashflow(user_id, user_txns, user_cf, roundup_engine)
    
    extra_cols = ['actual_credit_score', 'predicted_credit_score', 'prediction_error']
    user_feat_clean = user_feat.drop(columns=[c for c in extra_cols if c in user_feat.columns])
    X_user = credit_model.prepare_features(user_feat_clean)
    explanation = credit_model.explain_prediction(X_user, 0)
    
    nudge_engine = NudgeEngine()
    risk_assessment = nudge_engine.assess_user_risk(user_info.to_dict(), user_feat.iloc[0].to_dict())
    nudges = nudge_engine.recommend_for_user(user_id, user_info.to_dict(), user_feat.iloc[0].to_dict())
    
    return {
        'user_profile': user_info.to_dict(),
        'cashflow_simulation': cashflow_sim,
        'credit_score': {
            'predicted': explanation['prediction'],
            'explanation': explanation
        },
        'risk_assessment': risk_assessment,
        'insurance_recommendations': [
            {
                'product': n.product.name,
                'type': n.product.type.value,
                'premium': n.product.monthly_premium,
                'coverage': n.product.coverage_amount,
                'priority': n.priority_score,
                'urgency': n.urgency,
                'message': n.message
            }
            for n in nudges
        ],
        'savings_summary': {
            'total_transactions': len(user_txns),
            'total_spent': user_txns['amount'].sum(),
            'total_roundup': user_txns['round_up_amount'].sum(),
            'roundup_rate': user_txns['round_up_amount'].sum() / max(1, user_txns['amount'].sum())
        }
    }


def main():
    print("="*60)
    print("MICRO-INVESTMENT PLATFORM - AI ENGINE PIPELINE")
    print("="*60)
    
    print("\nLoading training data...")
    train_data = load_data('data', 'train')
    print(f"  Users: {len(train_data['users'])}")
    print(f"  Transactions: {len(train_data['transactions'])}")
    print(f"  Cashflows: {len(train_data['cashflows'])}")
    
    print("\nLoading test data...")
    test_data = load_data('data', 'test')
    print(f"  Users: {len(test_data['users'])}")
    print(f"  Transactions: {len(test_data['transactions'])}")
    print(f"  Cashflows: {len(test_data['cashflows'])}")
    
    roundup_train = run_roundup_engine(train_data['transactions'], train_data['cashflows'])
    roundup_test = run_roundup_engine(test_data['transactions'], test_data['cashflows'])
    
    roundup_train.to_csv('data/train/roundup_processed.csv', index=False)
    roundup_test.to_csv('data/test/roundup_processed.csv', index=False)
    print("Saved roundup processed data")
    
    credit_model, test_features, shap_summary = run_credit_scoring(
        train_data, test_data, roundup_train, roundup_test
    )
    
    nudges_df = run_insurance_nudges(test_data, test_features)
    
    print("\n" + "="*60)
    print("SAMPLE USER REPORT")
    print("="*60)
    
    sample_user = test_data['users']['user_id'].iloc[0]
    report = generate_user_report(sample_user, test_data, roundup_test, credit_model, test_features)
    
    print(f"\nUser ID: {sample_user}")
    print(f"Age: {report['user_profile']['age']}, Income: BDT {report['user_profile']['monthly_income']:,.0f}/mo")
    print(f"Risk Tolerance: {report['user_profile']['risk_tolerance']}")
    print(f"Investment Goal: {report['user_profile']['investment_goal']}")
    print(f"\nPredicted Credit Score: {report['credit_score']['predicted']:.0f}")
    print(f"Reserve Ratio: {report['risk_assessment']['reserve_ratio']:.1%}")
    print(f"Risk Score: {report['risk_assessment']['risk_score']:.2f}")
    print(f"Triggers: {[t.value for t in report['risk_assessment']['triggers']]}")
    
    print("\nInsurance Recommendations:")
    for rec in report['insurance_recommendations']:
        print(f"  - {rec['product']} ({rec['urgency']}): BDT {rec['premium']}/mo")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nOutputs generated:")
    print("  - models/credit_model.joblib (trained XGBoost model)")
    print("  - outputs/insurance_nudges.csv (recommendations)")
    print("  - data/train/roundup_processed.csv (roundup transactions)")
    print("  - data/test/roundup_processed.csv (roundup transactions)")


if __name__ == '__main__':
    main()