import os
import logging
import json
import mlflow
from typing import Dict, Any
import mlflow.server
from lm_eval import evaluator
from lm_eval.tasks import get_task_dict
from lm_eval.models.openai_completions import LocalCompletionsAPI

class AIBEModelEvaluator:
    def __init__(
        self,
        model_name: str = "babbage-002",
        num_fewshot: int = 5,
        batch_size: int = 4,
        openai_api_key: str = None
    ):
        """
        Initialize the AIBE Model Evaluator
        """
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # API Key validation
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided")

        # Set environment variable for OpenAI API key
        os.environ['OPENAI_API_KEY'] = self.openai_api_key

        # Evaluation parameters
        self.model_name = model_name
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size

        # MLflow setup
        mlflow.set_experiment("AIBE_Model_Evaluation")
        # mlflow.server.run(host='127.0.0.1', port=5000)
    def prepare_model(self):
        """
        Prepare the OpenAI model for evaluation
        """
        return LocalCompletionsAPI(
            model=self.model_name,
            base_url="https://api.openai.com/v1/completions",
            tokenizer="gpt2",
            tokenizer_backend='huggingface',
            tokenized_requests=True,
            add_bos_token=False,
            num_concurrent=1,
            max_retries=3,
            timeout=30
        )

    def clean_metric_name(self, metric_name: str) -> str:
        """
        Clean metric names to remove invalid characters
        """
        return ''.join(
            char for char in str(metric_name)
            if char.isalnum() or char in ['_', '-', '.', ' ', ':', '/']
        )

    def run_evaluation(self):
        """
        Run comprehensive model evaluation
        """
        with mlflow.start_run() as run:
            # Log run metadata
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("dataset", "AIBE")
            mlflow.set_tag("evaluation_type", "multiple_choice")

            # Log evaluation parameters
            mlflow.log_params({
                "model": self.model_name,
                "num_fewshot": self.num_fewshot,
                "batch_size": self.batch_size
            })

            try:
                # Prepare model
                model = self.prepare_model()

                # Create a custom task configuration
                task_dict = get_task_dict(["aibe"])

                # Modify the task's configuration
                if "aibe" in task_dict:
                    task_dict["aibe"].config.update({
                        "dataset_path": "ParimalThakre/evaluation_test",
                        "dataset_name": None,
                        "dataset_kwargs": {},
                        "doc_to_text": "Question: {{question}}\nOptions: {{options}}\nAnswer: ",
                        "doc_to_target": "{{correct_option}}",
                        "doc_to_choice": ["a", "b", "c", "d"],
                        "output_type": "multiple_choice"
                    })

                # Run evaluation
                results = evaluator.simple_evaluate(
                    model=model,
                    tasks=["aibe"],
                    num_fewshot=self.num_fewshot
                )

                # Extract and log metrics
                if results and 'results' in results and 'aibe' in results['results']:
                    metrics = results['results']['aibe']

                    # Log detailed metrics
                    for metric_name, metric_value in metrics.items():
                        clean_name = self.clean_metric_name(f"aibe_{metric_name}")

                        if isinstance(metric_value, (int, float)):
                            try:
                                mlflow.log_metric(clean_name, float(metric_value))
                            except Exception as log_error:
                                self.logger.warning(f"Could not log metric {clean_name}: {log_error}")

                    # Log primary accuracy
                    accuracy = metrics.get('acc', None)
                    if accuracy is not None:
                        try:
                            mlflow.log_metric("aibe_accuracy", float(accuracy))
                            self.logger.info(f"AIBE Assessment Accuracy: {accuracy}")
                        except Exception as acc_error:
                            self.logger.warning(f"Could not log accuracy: {acc_error}")

                # Save results to JSON
                results_path = 'aibe_evaluation_results.json'
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)

                # Log results file to MLflow
                mlflow.log_artifact(results_path)

                # Log run ID for reference
                run_id = run.info.run_id
                self.logger.info(f"MLflow Run ID: {run_id}")

                return results

            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                mlflow.log_param("evaluation_error", str(e))
                raise

def main():
    """
    Main entry point for the AIBE evaluation script
    """
    # Create and run evaluator
    evaluator_instance = AIBEModelEvaluator(
        model_name="babbage-002",
        num_fewshot=5,
        batch_size=4
    )

    # Run evaluation
    evaluator_instance.run_evaluation()

    # Minimal console output
    print("\nEvaluation completed. Check MLflow dashboard for detailed results.")

if __name__ == "__main__":
    main()
