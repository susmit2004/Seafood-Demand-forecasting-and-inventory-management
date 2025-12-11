import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.reference_stats = self._calculate_statistics(reference_data)
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical properties of data"""
        stats = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            stats[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'skew': data[column].skew(),
                'kurtosis': data[column].kurtosis()
            }
        return stats
    
    def detect_drift(self, current_data: pd.DataFrame, threshold: float = 0.05) -> Dict:
        """Detect data drift between reference and current data"""
        drift_report = {}
        current_stats = self._calculate_statistics(current_data)
        
        for column in self.reference_stats.keys():
            if column not in current_stats:
                continue
                
            ref_stats = self.reference_stats[column]
            curr_stats = current_stats[column]
            
            # KS test for distribution similarity
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[column].dropna(),
                current_data[column].dropna()
            )
            
            # Mean shift detection
            mean_shift = abs(ref_stats['mean'] - curr_stats['mean']) / ref_stats['std']
            
            drift_report[column] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < threshold,
                'mean_shift': mean_shift,
                'severity': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
            }
        
        return drift_report
    
    def generate_alert(self, drift_report: Dict) -> None:
        """Generate alerts for significant drift"""
        high_drift_columns = [
            col for col, report in drift_report.items() 
            if report['severity'] == 'high'
        ]
        
        if high_drift_columns:
            logger.warning(f"High data drift detected in columns: {high_drift_columns}")
            # Here you would integrate with your alerting system (email, Slack, etc.)