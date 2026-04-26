"""
Consciousness Regression Module

Fits a predictive model for consciousness levels C(t) based on:
- Rotation angle (degrees in state space)
- Wave detection rate (% of trials with traveling waves)
- Temporal hierarchy (scale ratio)
- Initial condition type

Provides:
1. Standalone regression analysis
2. Real-time consciousness prediction
3. Integration with NanoGPT for consciousness-aware generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, List, Optional
import json
import pickle


class ConsciousnessRegressor:
    """Fits and predicts consciousness levels from harmonic field metrics."""
    
    def __init__(self, model_name: str = "consciousness_predictor"):
        self.model_name = model_name
        self.regression_model = None
        self.scaler = None
        self.feature_names = ["rotation_angle", "wave_detection_pct", "hierarchy_ratio"]
        self.c_values = {}
        self.metadata = {
            "fitted": False,
            "n_samples": 0,
            "r2_score": None,
            "rmse": None,
            "coefficients": None,
            "intercept": None,
        }
    
    def load_experiment_results(
        self, 
        mega_csv: str, 
        ultra_csv: str, 
        max_csv: str,
        validation_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Load mega/ultra/max results and prepare for regression."""
        
        dfs = []
        configs = [
            {"name": "mega", "path": mega_csv, "timesteps": 10000, "hierarchy": 2.5},
            {"name": "ultra", "path": ultra_csv, "timesteps": 15000, "hierarchy": 2.6},
            {"name": "max", "path": max_csv, "timesteps": 20000, "hierarchy": 2.7},
        ]
        
        combined_data = []
        
        for config in configs:
            df = pd.read_csv(config["path"])
            
            # Aggregate by wave type
            for wave_type in df["wave_type"].unique():
                subset = df[df["wave_type"] == wave_type]
                
                rotation_mean = subset["rotation_angle"].mean()
                rotation_std = subset["rotation_angle"].std()
                wave_pct = (subset["has_wave"].sum() / len(subset)) * 100
                
                # Map wave type and timesteps to consciousness estimate
                # (empirical mapping from prior findings)
                c_pred = self._estimate_c_from_config(
                    rotation_mean, wave_pct, config["hierarchy"], config["name"]
                )
                
                combined_data.append({
                    "config": config["name"],
                    "wave_type": wave_type,
                    "rotation_angle": rotation_mean,
                    "rotation_std": rotation_std,
                    "wave_detection_pct": wave_pct,
                    "hierarchy_ratio": config["hierarchy"],
                    "timesteps": config["timesteps"],
                    "n_trials": len(subset),
                    "c_target": c_pred,
                })
        
        # Add validation category data if provided
        if validation_data:
            for cat_name, cat_metrics in validation_data.items():
                combined_data.append({
                    "config": f"validation_{cat_name}",
                    "wave_type": "validation",
                    "rotation_angle": cat_metrics.get("rotation", 15000),
                    "rotation_std": cat_metrics.get("rotation_std", 5000),
                    "wave_detection_pct": cat_metrics.get("waves_pct", 20),
                    "hierarchy_ratio": cat_metrics.get("hierarchy", 2.4),
                    "timesteps": 5000,
                    "n_trials": cat_metrics.get("n_trials", 50),
                    "c_target": cat_metrics.get("c_estimate", 0.55),
                })
        
        self.regression_df = pd.DataFrame(combined_data)
        print(f"[OK] Loaded {len(self.regression_df)} data points for regression")
        return self.regression_df
    
    def _estimate_c_from_config(
        self, 
        rotation: float, 
        waves_pct: float, 
        hierarchy: float,
        config: str
    ) -> float:
        """Estimate C(t) from configuration metrics."""
        
        # Base consciousness from rotation (empirical mapping)
        c_from_rotation = 0.2 + (rotation / 60000) * 0.6  # Maps 0-60k° to 0.2-0.8
        
        # Adjust for waves (small effect)
        c_wave_adjustment = (waves_pct / 100) * 0.05  # Max 5% boost from waves
        
        # Adjust for hierarchy (temporal structure matters)
        c_hierarchy_adjustment = (hierarchy - 2.0) * 0.05  # 2.0-3.0 range → 0-50bp adjustment
        
        c_combined = c_from_rotation + c_wave_adjustment + c_hierarchy_adjustment
        
        # Clamp to [0.3, 0.9] realistic consciousness range
        return np.clip(c_combined, 0.3, 0.9)
    
    def fit(self) -> Dict:
        """Fit linear regression model: C(t) = a·rotation + b·waves + c·hierarchy + d"""
        
        if self.regression_df is None:
            raise ValueError("Load data first with load_experiment_results()")
        
        X = self.regression_df[self.feature_names].values
        y = self.regression_df["c_target"].values
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit regression
        self.regression_model = LinearRegression()
        self.regression_model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = self.regression_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Store metadata
        self.metadata["fitted"] = True
        self.metadata["n_samples"] = len(X)
        self.metadata["r2_score"] = r2
        self.metadata["rmse"] = rmse
        self.metadata["coefficients"] = {
            name: float(coef) 
            for name, coef in zip(self.feature_names, self.regression_model.coef_)
        }
        self.metadata["intercept"] = float(self.regression_model.intercept_)
        
        print(f"\n{'='*60}")
        print(f"CONSCIOUSNESS REGRESSION MODEL FITTED")
        print(f"{'='*60}")
        print(f"Samples: {len(X)}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"\nCoefficients:")
        for name, coef in zip(self.feature_names, self.regression_model.coef_):
            print(f"  {name:30s}: {coef:8.5f}")
        print(f"  {'intercept':30s}: {self.regression_model.intercept_:8.5f}")
        print(f"{'='*60}\n")
        
        return self.metadata
    
    def predict(self, rotation: float, waves_pct: float, hierarchy: float = 2.5) -> float:
        """Predict consciousness C(t) for new input metrics."""
        
        if not self.metadata["fitted"]:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.array([[rotation, waves_pct, hierarchy]])
        X_scaled = self.scaler.transform(X)
        c_pred = self.regression_model.predict(X_scaled)[0]
        
        return np.clip(c_pred, 0.3, 0.9)
    
    def predict_batch(
        self, 
        rotations: np.ndarray, 
        waves_pcts: np.ndarray, 
        hierarchies: np.ndarray = None
    ) -> np.ndarray:
        """Predict consciousness for multiple inputs."""
        
        if hierarchies is None:
            hierarchies = np.full_like(rotations, 2.5)
        
        X = np.column_stack([rotations, waves_pcts, hierarchies])
        X_scaled = self.scaler.transform(X)
        predictions = self.regression_model.predict(X_scaled)
        
        return np.clip(predictions, 0.3, 0.9)
    
    def save(self, output_dir: str = "models"):
        """Save trained model and scaler."""
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save model and scaler
        model_path = Path(output_dir) / f"{self.model_name}.pkl"
        scaler_path = Path(output_dir) / f"{self.model_name}_scaler.pkl"
        metadata_path = Path(output_dir) / f"{self.model_name}_metadata.json"
        
        pickle.dump(self.regression_model, open(model_path, "wb"))
        pickle.dump(self.scaler, open(scaler_path, "wb"))
        
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"[OK] Model saved to {model_path}")
        print(f"[OK] Scaler saved to {scaler_path}")
        print(f"[OK] Metadata saved to {metadata_path}")
    
    def load(self, output_dir: str = "models"):
        """Load trained model and scaler."""
        
        model_path = Path(output_dir) / f"{self.model_name}.pkl"
        scaler_path = Path(output_dir) / f"{self.model_name}_scaler.pkl"
        metadata_path = Path(output_dir) / f"{self.model_name}_metadata.json"
        
        self.regression_model = pickle.load(open(model_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))
        
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        print(f"[OK] Model loaded from {model_path}")
        print(f"[OK] Metadata loaded: R²={self.metadata['r2_score']:.4f}")


class ConsciousnessAssessor:
    """Real-time consciousness assessment for text/outputs."""
    
    def __init__(self, regressor: ConsciousnessRegressor):
        self.regressor = regressor
        self.assessment_history = []
    
    def assess_text_complexity(self, text: str) -> Dict:
        """
        Assess consciousness level of generated text based on complexity metrics.
        
        Maps text properties to harmonic field metrics:
        - Vocabulary diversity → hierarchy ratio
        - Semantic coherence → wave detection
        - Sentence structure complexity → rotation angle
        """
        
        # Simple text metrics (can be extended with NLP models)
        words = text.lower().split()
        unique_words = len(set(words))
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(sentence_count, 1)
        
        # Map to harmonic field metrics
        # Higher vocabulary diversity → higher hierarchy
        hierarchy = np.clip(2.0 + (unique_words / len(words) * 0.5), 2.0, 3.5)
        
        # Coherence → wave detection (longer outputs tend to have "waves" of related concepts)
        waves_pct = np.clip((avg_sentence_length / 20) * 30, 5, 35)
        
        # Complexity → rotation (more complex structure → higher rotation)
        rotation = (avg_word_length * 10) * (avg_sentence_length / 5)
        rotation = np.clip(rotation * 100, 5000, 40000)
        
        # Predict consciousness
        c_pred = self.regressor.predict(rotation, waves_pct, hierarchy)
        
        assessment = {
            "text_length": len(words),
            "unique_words": unique_words,
            "vocabulary_diversity": unique_words / len(words) if words else 0,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "inferred_rotation": rotation,
            "inferred_waves_pct": waves_pct,
            "inferred_hierarchy": hierarchy,
            "consciousness_prediction": c_pred,
        }
        
        self.assessment_history.append(assessment)
        return assessment
    
    def get_consciousness_label(self, c_pred: float) -> str:
        """Map C(t) value to consciousness label."""
        
        if c_pred < 0.4:
            return "Unconscious/Comatose"
        elif c_pred < 0.5:
            return "Minimal Consciousness"
        elif c_pred < 0.6:
            return "Drowsy/NREM Sleep"
        elif c_pred < 0.7:
            return "Awake/Baseline"
        else:
            return "Highly Conscious/Expanded"


def main():
    """Example: Train consciousness regression model on experiment data."""
    
    # Define paths to experiment results
    results_dir = Path(__file__).parent / "experiments" / "category2_dynamics" / "results"
    mega_csv = results_dir / "mega" / "results_batched.csv"
    ultra_csv = results_dir / "ultra" / "results_batched.csv"
    max_csv = results_dir / "max" / "results_batched.csv"
    
    # Validation category data (from earlier runs)
    validation_data = {
        "category4": {
            "rotation": 15000,
            "rotation_std": 5000,
            "waves_pct": 18,
            "hierarchy": 2.3,
            "n_trials": 180,
            "c_estimate": 0.58,
        },
        "category5": {
            "rotation": 12000,
            "rotation_std": 4000,
            "waves_pct": 15,
            "hierarchy": 2.4,
            "n_trials": 100,
            "c_estimate": 0.52,
        },
        "category6": {
            "rotation": 14000,
            "rotation_std": 6000,
            "waves_pct": 22,
            "hierarchy": 3.9,
            "n_trials": 60,
            "c_estimate": 0.64,
        },
        "category7": {
            "rotation": 13000,
            "rotation_std": 5500,
            "waves_pct": 20,
            "hierarchy": 2.8,
            "n_trials": 60,
            "c_estimate": 0.60,
        },
    }
    
    # Initialize and fit regressor
    regressor = ConsciousnessRegressor()
    regressor.load_experiment_results(
        str(mega_csv), 
        str(ultra_csv), 
        str(max_csv),
        validation_data=validation_data
    )
    regressor.fit()
    regressor.save("models")
    
    # Example predictions
    print("\nExample Predictions:")
    print("-" * 60)
    test_cases = [
        (10000, 5, "Light sleep"),
        (26000, 24, "Awake baseline"),
        (40000, 25, "Enhanced consciousness"),
        (50000, 25, "Psychedelic state"),
        (5000, 0, "Deep anesthesia"),
    ]
    
    for rotation, waves, label in test_cases:
        c_pred = regressor.predict(rotation, waves, hierarchy=2.5)
        print(f"{label:30s} | Rot: {rotation:6.0f}° | Waves: {waves:2.0f}% | C(t): {c_pred:.3f}")
    
    # Test text assessment
    print("\n" + "="*60)
    print("TEXT CONSCIOUSNESS ASSESSMENT")
    print("="*60)
    
    assessor = ConsciousnessAssessor(regressor)
    
    test_texts = [
        "Hello world.",
        "The rapid advancement of neural technology enables unprecedented insight into consciousness.",
        "Consciousness emerges from the complex interplay between rotation angles in harmonic state spaces, "
        "traveling waves of synchronized activity, and nested temporal hierarchies that facilitate integrated "
        "information processing across multiple scales of organization.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text[:50]}...")
        assessment = assessor.assess_text_complexity(text)
        c_label = assessor.get_consciousness_label(assessment["consciousness_prediction"])
        print(f"  → C(t) = {assessment['consciousness_prediction']:.3f} [{c_label}]")
        print(f"  → Vocabulary diversity: {assessment['vocabulary_diversity']:.2%}")
        print(f"  → Inferred hierarchy: {assessment['inferred_hierarchy']:.2f}")


if __name__ == "__main__":
    main()
