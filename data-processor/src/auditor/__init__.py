from typing import List, Dict, Any
from .schema import SchemaAnalyzer
from .duplicates import DuplicateAndMissingAnalyzer
from .stats import StatsAnalyzer
from .quality import QualityAnalyzer
from .privacy import PrivacyAnalyzer
from .scorer import ScoringEngine

class DatasetAuditor:
    def __init__(self):
        self.analyzers = [
            SchemaAnalyzer(),
            DuplicateAndMissingAnalyzer(),
            StatsAnalyzer(),
            QualityAnalyzer(),
            PrivacyAnalyzer()
        ]
        self.scorer = ScoringEngine()
        
    def audit(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        report = {}
        for analyzer in self.analyzers:
            metrics = analyzer.analyze(dataset)
            report.update(metrics)
            
        # Final scoring and recommendations
        report = self.scorer.evaluate(report)
        return report

auditor = DatasetAuditor()
