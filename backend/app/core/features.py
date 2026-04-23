"""
Feature Engineering Pipeline
=============================
Every feature has a business reason — be ready to explain each in interviews.

Key features:
1. Velocity      — txns per 1h/24h (card testing pattern detection)
2. Amount zscore — how unusual is this amount vs cardholder history
3. Time anomaly  — is this outside user's normal hours
4. Merchant risk — historical fraud rate for this MCC code
5. Geo anomaly   — foreign merchant flag
6. Card newness  — new cards have higher fraud risk
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RawTransaction:
    transaction_id: str
    card_id: str
    merchant_id: str
    merchant_category_code: str
    amount: float
    transaction_type: str
    timestamp: datetime
    merchant_country: str
    is_online: bool
    device_id: Optional[str] = None


class FeatureEngineer:
    """
    Transforms raw transactions into ML-ready feature vectors.
    Interview: In production use a Feature Store (Feast/Tecton)
    to guarantee train/serve feature consistency.
    """

    # MCC codes with high historical fraud rates in SG fintech
    HIGH_RISK_MCC = {"5912","5999","7995","5094","4829","6051"}
    # Odd hours (SG UTC+8): midnight to 6am
    ODD_HOURS = set(range(0, 6))

    def __init__(self):
        self._card_history: dict[str, list[dict]] = {}

    def engineer(self, txn: RawTransaction) -> dict:
        """Raw transaction → feature dict. Must match training features exactly."""
        history = self._card_history.get(txn.card_id, [])
        features = {
            # Amount features
            "amount":                txn.amount,
            "amount_log":            np.log1p(txn.amount),
            "amount_zscore":         self._zscore(txn.amount, history),
            "amount_percentile":     self._percentile(txn.amount, history),
            "is_round_amount":       float(txn.amount % 10 == 0),
            "is_large_amount":       float(txn.amount > 5000),
            # Velocity features (key signal: card testing uses many small txns)
            "txn_count_1h":          self._count_recent(history, txn.timestamp, 1),
            "txn_count_24h":         self._count_recent(history, txn.timestamp, 24),
            "txn_count_7d":          self._count_recent(history, txn.timestamp, 168),
            "amount_sum_1h":         self._sum_recent(history, txn.timestamp, 1),
            "amount_sum_24h":        self._sum_recent(history, txn.timestamp, 24),
            "unique_merchants_24h":  self._unique_merchants(history, txn.timestamp, 24),
            # Time features
            "hour_of_day":           txn.timestamp.hour,
            "day_of_week":           txn.timestamp.weekday(),
            "is_weekend":            float(txn.timestamp.weekday() >= 5),
            "is_odd_hours":          float(txn.timestamp.hour in self.ODD_HOURS),
            "time_since_last_min":   self._time_since_last(history, txn.timestamp),
            # Merchant features
            "is_high_risk_mcc":      float(txn.merchant_category_code in self.HIGH_RISK_MCC),
            "is_foreign_merchant":   float(txn.merchant_country != "SG"),
            "is_online":             float(txn.is_online),
            # Card history features
            "card_txn_count_total":  len(history),
            "card_is_new":           float(len(history) < 5),
            "card_avg_amount":       np.mean([h["amount"] for h in history]) if history else txn.amount,
            # Device
            "has_device_id":         float(txn.device_id is not None),
            "device_is_new":         self._device_is_new(txn, history),
        }
        self._update_history(txn)
        return features

    def _zscore(self, amount, history):
        if len(history) < 3: return 0.0
        amounts = [h["amount"] for h in history]
        return (amount - np.mean(amounts)) / (np.std(amounts) + 1e-9)

    def _percentile(self, amount, history):
        if len(history) < 3: return 50.0
        return float(np.mean(np.array([h["amount"] for h in history]) <= amount) * 100)

    def _count_recent(self, history, ts, hours):
        cutoff = ts.timestamp() - hours * 3600
        return sum(1 for h in history if h["ts"] >= cutoff)

    def _sum_recent(self, history, ts, hours):
        cutoff = ts.timestamp() - hours * 3600
        return sum(h["amount"] for h in history if h["ts"] >= cutoff)

    def _unique_merchants(self, history, ts, hours):
        cutoff = ts.timestamp() - hours * 3600
        return len(set(h["merchant_id"] for h in history if h["ts"] >= cutoff))

    def _time_since_last(self, history, ts):
        if not history: return 999.0
        last = max(h["ts"] for h in history)
        return (ts.timestamp() - last) / 60

    def _device_is_new(self, txn, history):
        if not txn.device_id or not history: return 0.0
        seen = {h.get("device_id") for h in history}
        return float(txn.device_id not in seen)

    def _update_history(self, txn):
        if txn.card_id not in self._card_history:
            self._card_history[txn.card_id] = []
        self._card_history[txn.card_id].append({
            "ts": txn.timestamp.timestamp(),
            "amount": txn.amount,
            "merchant_id": txn.merchant_id,
            "device_id": txn.device_id,
        })
        # Keep last 200 transactions per card
        if len(self._card_history[txn.card_id]) > 200:
            self._card_history[txn.card_id] = self._card_history[txn.card_id][-200:]
