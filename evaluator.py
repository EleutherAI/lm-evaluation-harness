import os
import logging
import json
import mlflow
from typing import Dict, Any
import mlflow.server
from lm_eval import evaluator
from lm_eval.tasks import get_task_dict
import datasets
from dotenv import load_dotenv
import argilla as rg
from argilla import TextField, RatingQuestion, LabelQuestion, Dataset
import re
import copy

from azure_openai_wrapper import AzureOpenAIWrapper

class AIBEModelEvaluator:
    def __init__(
        self,
        model_name: str = "courteasy-ai-gpt4o",  # Default to GPT-4
        num_fewshot: int = 5,
        batch_size: int = 4,
        openai_api_key: str = None,
        huggingface_token: str = None,
        argilla_api_key: str = None,
        argilla_api_url: str = None,
        azure_openai_api_key: str = None,  # Add Azure OpenAI API key
        azure_openai_endpoint: str = None  # Add Azure OpenAI endpoint
    ):
        """
        Initialize the AIBE Model Evaluator
        """
        load_dotenv()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # API Key validation
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.azure_openai_api_key = azure_openai_api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_openai_endpoint = azure_openai_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')

        if not self.azure_openai_api_key or not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI API key and endpoint must be provided")

        self.huggingface_token = huggingface_token or os.getenv('HUGGINGFACE_TOKEN')
        if not self.huggingface_token:
            raise ValueError("Hugging Face token must be provided")

        # Argilla setup
        self.argilla_api_key = argilla_api_key or os.getenv('ARGILLA_API_KEY')
        self.argilla_api_url = argilla_api_url or os.getenv('ARGILLA_API_URL')

        if self.argilla_api_key and self.argilla_api_url:
            self.argilla_client = rg.Argilla(
                api_url=self.argilla_api_url,
                api_key=self.argilla_api_key
            )
            current_user = self.argilla_client.me
            user_id = current_user.id
            self.logger.info("Initialized Argilla client")

        os.environ['OPENAI_API_KEY'] = self.openai_api_key
        os.environ['HUGGINGFACE_TOKEN'] = self.huggingface_token

        self.model_name = model_name
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size

        mlflow.set_experiment("AIBE_Model_Evaluation")

    def create_argilla_dataset(self):
        """
        Create and configure Argilla dataset with separate fields for each option
        """
        try:
            dataset_name = "aibe_evaluation"

            # Delete dataset if it already exists
            try:
                existing_datasets = self.argilla_client.datasets.list()
                for dataset in existing_datasets:
                    if dataset.name == dataset_name:
                        self.logger.info(f"Deleting existing dataset: {dataset_name}")
                        self.argilla_client.datasets.delete(name=dataset_name)
                        self.logger.info(f"Deleted existing dataset: {dataset_name}")
            except Exception as e:
                self.logger.debug(f"Error checking existing datasets: {e}")

            # Create settings with individual fields for each option
            settings = rg.Settings(
                guidelines="AIBE evaluation results for legal multiple choice questions",
                fields=[
                    # Question field
                    rg.TextField(
                        name="question",
                        title="Question",
                        required=True,
                    ),
                    
                    # Individual option fields
                    rg.TextField(
                        name="option_a",
                        title="Option A",
                        required=True,
                    ),
                    rg.TextField(
                        name="option_b",
                        title="Option B",
                        required=True,
                    ),
                    rg.TextField(
                        name="option_c",
                        title="Option C",
                        required=True,
                    ),
                    rg.TextField(
                        name="option_d",
                        title="Option D",
                        required=True,
                    ),
                    
                    # Model response field
                    rg.TextField(
                        name="model_response",
                        title="Model Response",
                        required=True,
                    ),
                    
                    # Target response field
                    rg.TextField(
                        name="target_response",
                        title="Target Response (Correct Answer)",
                        required=True,
                    ),
                ],
                questions=[
                    # Validation question
                    rg.LabelQuestion(
                        name="validate",
                        title="Validate",
                        labels=["Yes", "No"],
                        required=True,
                    ),
                    
                    # Comments field for evaluator feedback
                    rg.TextField(
                        name="comments",
                        title="Comments",
                        required=False,
                        use_markdown=True,
                    ),
                ],
            )

            # Create dataset with settings
            dataset = rg.Dataset(
                name=dataset_name,
                workspace="default",
                settings=settings,
            )
            
            dataset.create()
            self.logger.info(f"Created Argilla dataset: {dataset_name}")
            return dataset

        except Exception as e:
            self.logger.error(f"Failed to create Argilla dataset: {e}")
            raise

    def extract_individual_options(self, raw_options):
        """
        Extract individual options from the raw options string
        
        Args:
            raw_options: The raw options string from the dataset
        Returns:
            Tuple of (option_a, option_b, option_c, option_d)
        """
        # Default empty options
        options = {
            'a': "No option A provided",
            'b': "No option B provided",
            'c': "No option C provided",
            'd': "No option D provided"
        }
        
        # Universal pattern for all option types
        pattern = r'([a-d][\)\.])\s*(.*?)(?=[a-d][\)\.]|$)'
        matches = re.findall(pattern, raw_options, re.DOTALL)
        
        if matches:
            for marker, text in matches:
                option_letter = marker[0].lower()
                options[option_letter] = text.strip()
        else:
            # For comma-separated options, use alternative approach
            if ',' in raw_options:
                parts = raw_options.split(',')
                current_letter = None
                current_text = ""
                
                for part in parts:
                    part = part.strip()
                    
                    # Check if this part starts with an option marker
                    if part and part[0].lower() in ['a', 'b', 'c', 'd'] and len(part) > 1 and part[1] in [')', '.']:
                        # Save previous option if it exists
                        if current_letter:
                            options[current_letter] = current_text.strip()
                        
                        # Start new option
                        current_letter = part[0].lower()
                        current_text = part[2:].strip()  # Remove the "a) " part
                    else:
                        # This is a continuation of the current option
                        if current_letter:
                            current_text += ", " + part
                
                # Save the last option
                if current_letter:
                    options[current_letter] = current_text.strip()
        
        # Return as a tuple in the expected order
        return (options['a'], options['b'], options['c'], options['d'])
                
    def normalize_model_response(self, response):
        """
        Improved method to normalize the model's response to extract a valid option (a, b, c, or d)
        
        Args:
            response: The raw response from the model
        Returns:
            The normalized option (a, b, c, or d) or None if no valid option found
        """
        if not response:
            return None
            
        # Convert response to lowercase and strip whitespace
        response = response.lower().strip()
        
        # Check for exact single letter matches first
        if response in ['a', 'b', 'c', 'd']:
            return response
            
        # Check for patterns like "option a", "the answer is a", etc.
        prefixes = ['option ', 'answer ', 'answer: ', 'the answer is ', 'choice ']
        for prefix in prefixes:
            if response.startswith(prefix):
                # Extract the first letter after the prefix
                remainder = response[len(prefix):].strip()
                if remainder and remainder[0] in ['a', 'b', 'c', 'd']:
                    return remainder[0]
        
        # Look for answers within the text with common indicators
        indicators = [') ', '. ', ': ']
        for indicator in indicators:
            for option in ['a', 'b', 'c', 'd']:
                if f"{option}{indicator}" in response:
                    return option
        
        # Final check - scan for any a, b, c, d in the response
        # prioritizing those that appear at the beginning
        option_positions = []
        for option in ['a', 'b', 'c', 'd']:
            pos = response.find(option)
            if pos >= 0:
                option_positions.append((pos, option))
        
        if option_positions:
            # Sort by position (lowest/earliest first)
            option_positions.sort()
            return option_positions[0][1]
        
        return None

    def log_to_argilla(self, results):
        """
        Log evaluation results to Argilla with separate fields for each option
        """
        try:
            dataset = self.create_argilla_dataset()
            self.logger.info("Created Argilla dataset successfully")

            records = []

            # Validate results structure
            if not isinstance(results, dict) or 'samples' not in results or 'aibe' not in results['samples']:
                self.logger.error("Invalid results format")
                return

            self.logger.info(f"Total samples to process: {len(results['samples']['aibe'])}")

            for i, sample in enumerate(results['samples']['aibe']):
                try:
                    # Extract and validate document info
                    doc = sample.get('doc', {})
                    if not isinstance(doc, dict):
                        self.logger.warning(f"Sample {i}: Invalid doc format")
                        continue

                    # Extract required fields
                    question = doc.get('question', '')
                    raw_options = doc.get('options', '')
                    correct_option = doc.get('correct_option', '')
                    raw_response = sample.get('raw_model_response', 'No raw response available')
                    
                    # Extract individual options from the raw options string
                    option_a, option_b, option_c, option_d = self.extract_individual_options(raw_options)
                    
                    # Extract model's selected option
                    model_option = self.normalize_model_response(raw_response)
                    
                    # Format model response to display the selection clearly
                    formatted_model_response = f"Option {model_option.upper()}" if model_option else "No clear option selected"
                    
                    # Format the target response with the correct letter and full answer if available
                    correct_answer = doc.get('correct_answer', '')
                    if correct_answer:
                        formatted_target_response = f"Option {correct_option.upper()}: {correct_answer}"
                    else:
                        formatted_target_response = f"Option {correct_option.upper()}"
                    
                    # Check if model's response is correct
                    is_correct = (model_option == correct_option.lower())
                    validation_value = "Yes" if is_correct else "No"
                            
                    # Create record with the EXACT same field names as defined in create_argilla_dataset
                    try:
                        record = rg.Record(
                            fields={
                                "question": question,
                                "option_a": option_a,
                                "option_b": option_b,
                                "option_c": option_c,
                                "option_d": option_d,
                                "model_response": formatted_model_response,
                                "target_response": formatted_target_response,
                            },
                            responses=[
                                rg.Response(
                                    value=validation_value,
                                    user_id=str(self.argilla_client.me.id),
                                    question_name="validate"
                                ),
                                # Comments field will be empty initially and filled by evaluators
                                rg.Response(
                                    value="",
                                    user_id=str(self.argilla_client.me.id),
                                    question_name="comments"
                                )
                            ]
                        )
                        records.append(record)
                        self.logger.info(f"Sample {i}: Successfully created record")
                        
                    except Exception as record_error:
                        self.logger.error(f"Sample {i}: Failed to create record: {str(record_error)}")
                        continue

                except Exception as sample_error:
                    self.logger.error(f"Sample {i}: Failed to process sample: {str(sample_error)}")
                    continue

            # Try to log one record at a time first
            if records:
                success_count = 0
                for i, record in enumerate(records):
                    try:
                        # Log one record at a time
                        dataset.records.log([record])
                        success_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to log individual record {i}: {e}")
                
                self.logger.info(f"Successfully logged {success_count} out of {len(records)} records")
                
                if success_count == 0:
                    self.logger.warning("No records were successfully logged to Argilla")
            else:
                self.logger.warning("No valid records to log to Argilla")

        except Exception as e:
            self.logger.error(f"Failed to log results to Argilla: {str(e)}")
            raise

    def format_options(self, raw_options):
        """
        Format options to ensure each option is on a separate line
        
        Args:
            raw_options: The raw options string from the dataset
        Returns:
            A formatted string with each option on a separate line
        """
        # If already formatted with line breaks, return as is
        if '\n' in raw_options:
            return raw_options
        
        # Try to format based on common patterns
        formatted = ""
        
        # Handle comma-separated options
        if ',' in raw_options:
            parts = raw_options.split(',')
            for part in parts:
                part = part.strip()
                if part.startswith('a)') or part.startswith('b)') or part.startswith('c)') or part.startswith('d)'):
                    formatted += part + "\n"
                elif '.' in part and part[0].lower() in ['a', 'b', 'c', 'd']:
                    formatted += part + "\n"
        # Handle options with clear markers
        elif raw_options.count('a)') == 1 and raw_options.count('b)') == 1:
            # Extract each option using regex
            pattern = r'([a-d]\).*?)(?=[a-d]\)|$)'
            matches = re.findall(pattern, raw_options, re.DOTALL)
            for match in matches:
                formatted += match.strip() + "\n"
        # Fallback to simple splitting
        else:
            options = ["a) ", "b) ", "c) ", "d) "]
            prev_idx = 0
            for option in options:
                idx = raw_options.find(option, prev_idx)
                if idx >= 0:
                    next_option_idx = len(raw_options)
                    for next_opt in options:
                        next_idx = raw_options.find(next_opt, idx + len(option))
                        if next_idx > idx:
                            next_option_idx = min(next_option_idx, next_idx)
                    
                    formatted += raw_options[idx:next_option_idx].strip() + "\n"
                    prev_idx = next_option_idx
        
        # If formatting didn't work, use the original
        if not formatted:
            return raw_options
        
        return formatted.strip()
        
    def prepare_model(self):
        """
        Prepare the Azure OpenAI GPT-4 model for evaluation
        """
        return AzureOpenAIWrapper(
            azure_openai_api_key=self.azure_openai_api_key,
            azure_openai_endpoint=self.azure_openai_endpoint,
            deployment_name=self.model_name  # Use the deployment name
        )

    def clean_metric_name(self, metric_name: str) -> str:
        """
        Clean metric names to remove invalid characters
        """
        return ''.join(
            char for char in str(metric_name)
            if char.isalnum() or char in ['_', '-', '.', ' ', ':', '/']
        )

    def load_dataset(self):
        """
        Load the dataset from Hugging Face
        """
        self.logger.info("Loading dataset from Hugging Face Hub...")
        try:
            dataset = datasets.load_dataset(
                "ParimalThakre/evaluation_test",
                download_mode="force_redownload"
            )
            self.logger.info("Successfully loaded dataset from Hugging Face Hub")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def run_evaluation(self):
        """
        Run comprehensive model evaluation with direct answers (no few-shot prompting)
        """
        with mlflow.start_run() as run:
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("dataset", "AIBE")
            mlflow.set_tag("evaluation_type", "multiple_choice")

            mlflow.log_params({
                "model": self.model_name,
                "num_fewshot": 0,  # Set to 0 to indicate no few-shot examples
                "batch_size": self.batch_size
            })

            try:
                model = self.prepare_model()
                task_dict = get_task_dict(["aibe"])

                if "aibe" in task_dict:
                    # Update task configuration for direct answer generation
                    # No few-shot examples, just direct questions and answers
                    task_dict["aibe"].config.update({
                        "dataset_path": "ParimalThakre/evaluation_test",
                        "dataset_name": None,
                        "dataset_kwargs": {"token": self.huggingface_token},
                        "doc_to_text": """Question: {{question}}
                        Options:
                        {{options}}
                        
                        Answer with ONLY the letter (a, b, c, or d) of the correct option.
                        Answer: """,
                        "doc_to_target": "{{correct_option}}",
                        "doc_to_choice": ["a", "b", "c", "d"],
                        "output_type": "multiple_choice"
                    })
                    
                dataset = self.load_dataset()
                
                # Use num_fewshot=0 to disable few-shot examples
                results = evaluator.simple_evaluate(
                    model=model,
                    tasks=["aibe"],
                    num_fewshot=0  # No few-shot examples
                )
                
                # Extract and store raw responses if available
                if hasattr(model, 'raw_responses'):
                    # Add raw responses to the results
                    for i, sample in enumerate(results.get('samples', {}).get('aibe', [])):
                        sample_idx = i % len(model.raw_responses)
                        if sample_idx < len(model.raw_responses):
                            sample['raw_model_response'] = model.raw_responses[sample_idx]
                
                # Create a copy of results for the debug version with full raw responses
                debug_results = copy.deepcopy(results)
                
                # Create a sanitized version with only extracted options for the regular results
                if 'samples' in results and 'aibe' in results['samples']:
                    for sample in results['samples']['aibe']:
                        if 'raw_model_response' in sample:
                            # Extract just the option from the raw response
                            raw_response = sample['raw_model_response']
                            extracted_option = self.normalize_model_response(raw_response) if raw_response else None
                            # Replace raw response with just the extracted option
                            sample['raw_model_response'] = extracted_option if extracted_option else "No option extracted"
                
                # Log results to MLflow
                if results and 'results' in results and 'aibe' in results['results']:
                    metrics = results['results']['aibe']
                    for metric_name, metric_value in metrics.items():
                        clean_name = self.clean_metric_name(f"aibe_{metric_name}")
                        if isinstance(metric_value, (int, float)):
                            try:
                                mlflow.log_metric(clean_name, float(metric_value))
                                self.logger.info(f"Logged metric {clean_name}: {metric_value}")
                            except Exception as log_error:
                                self.logger.warning(f"Could not log metric {clean_name}: {log_error}")

                    accuracy = metrics.get('acc', None)
                    if accuracy is not None:
                        try:
                            mlflow.log_metric("aibe_accuracy", float(accuracy))
                            self.logger.info(f"AIBE Assessment Accuracy: {accuracy}")
                        except Exception as acc_error:
                            self.logger.warning(f"Could not log accuracy: {acc_error}")

                # Save results to JSON files
                results_path = 'aibe_evaluation_results.json'
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=4)
                mlflow.log_artifact(results_path)
                self.logger.info(f"Saved evaluation results with extracted options to {results_path}")
                
                debug_results_path = 'aibe_evaluation_debug_results.json'
                with open(debug_results_path, 'w') as f:
                    json.dump(debug_results, f, indent=4)
                mlflow.log_artifact(debug_results_path)
                self.logger.info(f"Saved debug evaluation results with full raw responses to {debug_results_path}")

                # Log to Argilla if configured
                if self.argilla_api_key and self.argilla_api_url:
                    # Use the debug_results for Argilla logging to preserve raw responses
                    self.log_to_argilla(debug_results)

                self.logger.info(f"MLflow Run ID: {run.info.run_id}")
                return results

            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                mlflow.log_param("evaluation_error", str(e))
                raise