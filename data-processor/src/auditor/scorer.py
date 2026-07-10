from typing import Dict, Any

class ScoringEngine:
    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        score = 100
        recommendations = []
        
        # Deduct points for schema issues
        if not metrics.get("schema_valid", True):
            score -= 40
            recommendations.append("CRITICAL: Schema is completely invalid for the inferred type.")
            
        if metrics.get("inconsistent_rows", 0) > 0:
            score -= 10
            recommendations.append(f"WARNING: {metrics['inconsistent_rows']} rows have inconsistent keys.")
            
        # Missing values
        missing_pct = metrics.get("missing_percentage", 0)
        if missing_pct > 0:
            score -= int(missing_pct * 0.5) # Deduct half point for every 1% missing
            recommendations.append(f"WARNING: Dataset has {missing_pct}% missing values. Consider cleaning.")
            
        # Duplicates
        dup_pct = metrics.get("duplicate_percentage", 0)
        if dup_pct > 0:
            score -= int(dup_pct * 0.5)
            if dup_pct > 10:
                recommendations.append(f"CRITICAL: High duplicate rate ({dup_pct}%). This will degrade fine-tuning quality.")
            else:
                recommendations.append(f"INFO: Removed {dup_pct}% exact duplicates.")
                
        # Privacy
        if metrics.get("has_pii"):
            score -= 20
            recommendations.append(f"CRITICAL: Detected PII ({metrics.get('pii_emails_detected')} emails, {metrics.get('pii_ips_detected')} IPs). Anonymization required.")
            
        # Quality
        special_ratio = metrics.get("special_char_ratio", 0)
        if special_ratio > 0.1:
            score -= 10
            recommendations.append(f"WARNING: Unusually high special character ratio ({special_ratio}). May indicate corrupted text.")
            
        score = max(0, score) # Cap at 0
        
        # Calculate grade
        if score >= 90: grade = "A (Excellent)"
        elif score >= 75: grade = "B (Good)"
        elif score >= 60: grade = "C (Fair)"
        else: grade = "D (Poor/Unusable)"
        
        metrics["health_score"] = score
        metrics["grade"] = grade
        metrics["recommendations"] = recommendations
        
        return metrics
