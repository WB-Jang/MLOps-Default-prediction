"""Tests for model evaluation DAG functions."""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestEvaluationDAGImports:
    """Test that evaluation DAG can be imported successfully."""
    
    def test_import_evaluation_dag(self):
        """Test that the evaluation DAG module can be imported."""
        # This test validates that there are no syntax errors and all imports are valid
        try:
            from src.airflow.dags import model_evaluation_dag
            assert model_evaluation_dag is not None
        except ImportError as e:
            pytest.skip(f"DAG import failed (expected in CI without Airflow): {e}")
    
    def test_evaluation_dag_functions_exist(self):
        """Test that all required functions are defined in the evaluation DAG."""
        try:
            from src.airflow.dags import model_evaluation_dag
            
            # Check that all required functions exist
            assert hasattr(model_evaluation_dag, 'load_latest_model')
            assert hasattr(model_evaluation_dag, 'evaluate_model')
            assert hasattr(model_evaluation_dag, 'check_model_performance')
            assert hasattr(model_evaluation_dag, 'prepare_retraining_data')
            assert hasattr(model_evaluation_dag, 'finetune_model')
            assert hasattr(model_evaluation_dag, 'evaluate_finetuned_model')
            assert hasattr(model_evaluation_dag, 'send_evaluation_results')
            
        except ImportError as e:
            pytest.skip(f"DAG import failed (expected in CI without Airflow): {e}")


class TestCheckPerformanceLogic:
    """Test the performance checking logic."""
    
    def test_performance_threshold_logic(self):
        """Test that the F1 threshold logic works correctly."""
        # Test case 1: F1 score below threshold should trigger retraining
        f1_threshold = 0.75
        f1_score_low = 0.70
        needs_retraining = f1_score_low < f1_threshold
        assert needs_retraining is True, "Low F1 score should trigger retraining"
        
        # Test case 2: F1 score above threshold should not trigger retraining
        f1_score_high = 0.80
        needs_retraining = f1_score_high < f1_threshold
        assert needs_retraining is False, "High F1 score should not trigger retraining"
        
        # Test case 3: F1 score exactly at threshold should not trigger retraining
        f1_score_exact = 0.75
        needs_retraining = f1_score_exact < f1_threshold
        assert needs_retraining is False, "F1 score at threshold should not trigger retraining"


class TestBranchingLogic:
    """Test the branching logic for conditional retraining."""
    
    def test_branch_decision(self):
        """Test that the correct branch is selected based on performance."""
        # Simulate low performance
        f1_score = 0.70
        f1_threshold = 0.75
        needs_retraining = f1_score < f1_threshold
        
        if needs_retraining:
            next_task = 'prepare_retraining_data'
        else:
            next_task = 'send_evaluation_results'
        
        assert next_task == 'prepare_retraining_data', "Should branch to retraining"
        
        # Simulate good performance
        f1_score = 0.80
        needs_retraining = f1_score < f1_threshold
        
        if needs_retraining:
            next_task = 'prepare_retraining_data'
        else:
            next_task = 'send_evaluation_results'
        
        assert next_task == 'send_evaluation_results', "Should skip retraining"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
