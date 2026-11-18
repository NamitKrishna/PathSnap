"""
PATHSNAP – Model Training Package
==================================
Candidate-level supervised map-matching models using group-aware
cross-validation (GroupKFold by trip_id).

Modules:
    data_loader_ml      – Load & merge feature CSVs with label validation
    leakage_guard       – Partition integrity checks
    cross_validation    – GroupKFold CV utilities
    models              – Model registry with scalers
    metrics             – Candidate & observation-level metrics
    trainer             – Main training loop
    bayesian_optimization – Optuna / GridSearchCV HPO
    feature_importance  – Importance plots
    predict             – Inference utilities
    run_training        – CLI entry point
"""
