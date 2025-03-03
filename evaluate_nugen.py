import os
import sys
import json
import logging
import tempfile
import glob
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import mlflow
import argilla as rg
from argilla import TextField, LabelQuestion, Dataset

# Import evaluation-specific modules
try:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table
    EVAL_MODULES_AVAILABLE = True
except ImportError:
    EVAL_MODULES_AVAILABLE = False
    logging.warning("Evaluation modules not available. Only logging functionality will work.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Fixed configuration values (can be changed directly in the script)
RESULTS_FILE = "results.json"
RESULTS_DIR = "results_nugen_test_results"
MODEL_NAME = "nugen"
MODEL_ARGS = {"model": "nugen-flash-instruct", "temperature": 0.0}
TASKS = ["aibe"]
NUM_FEWSHOT = 0
BATCH_SIZE = 5
SYSTEM_PROMPT = """You are a highly capable assistant with expertise in law, ethics, and reasoning.
You are taking the Australian Bar Exam, which consists of multiple-choice questions.
Each question will have four options (A, B, C, D).
Read each question carefully, consider all options, and select the single most appropriate answer.
Provide your answer by stating the letter of your chosen option (A, B, C, or D).
Do not explain your reasoning unless specifically asked."""
MLFLOW_TRACKING_URI = "http://localhost:5000/"
MLFLOW_EXPERIMENT = "LLM Evaluation - Nugen Models"
ARGILLA_DATASET = "aibe_evaluation"
ARGILLA_BATCH_SIZE = 5

def run_evaluation():
    """
    Run the model evaluation and return results
    """
    if not EVAL_MODULES_AVAILABLE:
        logger.error("Evaluation modules not available. Cannot run evaluation.")
        return None
        
    logger.info("Starting evaluation...")
    
    try:
        # Add system prompt to model args if provided
        model_args = MODEL_ARGS.copy()
        if SYSTEM_PROMPT:
            model_args["system_prompt"] = SYSTEM_PROMPT
            
        # Initialize task manager
        task_manager = TaskManager()
        
        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Run evaluation
        results = evaluator.simple_evaluate(
            model=MODEL_NAME,
            model_args=model_args,
            tasks=TASKS,
            num_fewshot=NUM_FEWSHOT,
            batch_size=BATCH_SIZE,
            task_manager=task_manager
        )
        
        if results is not None:
            # Print results
            print(make_table(results))
            
            # Save results to a JSON file
            results_file = os.path.join(RESULTS_DIR, RESULTS_FILE)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation completed and results saved to {results_file}")
            return results
        else:
            logger.error("Evaluation returned no results")
            return None
            
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return None

def init_argilla_client():
    """Initialize Argilla client from environment variables"""
    argilla_api_key = os.getenv('ARGILLA_API_KEY')
    argilla_api_url = os.getenv('ARGILLA_API_URL')

    if not argilla_api_key or not argilla_api_url:
        logger.error("Missing required environment variables: ARGILLA_API_KEY and/or ARGILLA_API_URL")
        return None

    try:
        logger.info(f"Connecting to Argilla at {argilla_api_url}...")
        argilla_client = rg.Argilla(
            api_url=argilla_api_url,
            api_key=argilla_api_key
        )
        current_user = argilla_client.me
        user_id = current_user.id
        logger.info(f"Successfully connected to Argilla as user {user_id}")
        return argilla_client
    except Exception as e:
        logger.error(f"Failed to connect to Argilla: {e}")
        return None

def create_argilla_dataset(argilla_client, dataset_name):
    """
    Create and configure Argilla dataset with separate fields for each option
    """
    try:
        if argilla_client is None:
            logger.error("Cannot create dataset: Argilla client is not initialized")
            return None
            
        # Create settings with individual fields for each option
        settings = rg.Settings(
            guidelines="AIBE evaluation results for legal multiple choice questions",
            fields=[
                # Question field
                TextField(
                    name="question",
                    title="Question",
                    required=True,
                ),
                
                # Individual option fields
                TextField(
                    name="option_a",
                    title="Option A",
                    required=True,
                ),
                TextField(
                    name="option_b",
                    title="Option B",
                    required=True,
                ),
                TextField(
                    name="option_c",
                    title="Option C",
                    required=True,
                ),
                TextField(
                    name="option_d",
                    title="Option D",
                    required=True,
                ),
                
                # Model response field
                TextField(
                    name="model_response",
                    title="Model Response",
                    required=True,
                ),
                
                # Target response field
                TextField(
                    name="target_response",
                    title="Target Response (Correct Answer)",
                    required=True,
                ),
                
            ],
            questions=[
                # Validation question
                LabelQuestion(
                    name="validate",
                    title="Validate",
                    labels=["Yes", "No"],
                    required=True,
                ),
                
                # Comments field for evaluator feedback
                TextField(
                    name="comments",
                    title="Comments",
                    required=False,
                    use_markdown=True,
                ),
            ],
        )

        # Create dataset with settings
        dataset = Dataset(
            name=dataset_name,
            workspace="default",
            settings=settings,
        )
        
        dataset.create()
        logger.info(f"Created Argilla dataset: {dataset_name}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to create Argilla dataset: {e}")
        return None

def log_results_to_argilla(argilla_client, results, dataset_name, batch_size=5):
    """
    Log evaluation results to Argilla dataset
    """
    try:
        if argilla_client is None:
            logger.error("Cannot log results: Argilla client is not initialized")
            return False
            
        # Check if dataset exists, create if not
        try:
            dataset = argilla_client.datasets.find(name=dataset_name)
            logger.info(f"Found existing dataset: {dataset_name}")
        except:
            logger.info(f"Dataset {dataset_name} not found, creating...")
            dataset = create_argilla_dataset(argilla_client, dataset_name)
            if dataset is None:
                logger.error("Failed to create dataset")
                return False
        
        # Process samples and log to Argilla
        if "samples" in results:
            records = []
            for task_name, task_samples in results["samples"].items():
                for i, sample in enumerate(task_samples):
                    try:
                        # Extract data from the sample
                        question = sample.get("doc", {}).get("question", "")
                        
                        # Extract options for AIBE format (option_a, option_b, etc.)
                        option_a = sample.get("doc", {}).get("option_a", "")
                        option_b = sample.get("doc", {}).get("option_b", "")
                        option_c = sample.get("doc", {}).get("option_c", "")
                        option_d = sample.get("doc", {}).get("option_d", "")
                        
                        # If options are not found in the expected format, try the choices array
                        if not (option_a or option_b or option_c or option_d):
                            options = sample.get("doc", {}).get("choices", [])
                            option_a = options[0] if len(options) > 0 else ""
                            option_b = options[1] if len(options) > 1 else ""
                            option_c = options[2] if len(options) > 2 else ""
                            option_d = options[3] if len(options) > 3 else ""
                        
                        # Extract model response and target
                        model_response = sample.get("filtered_resps", [""])[0] if sample.get("filtered_resps") else ""
                        if not model_response and "model_response" in sample:
                            model_response = sample.get("model_response", "")
                            
                        target_response = sample.get("target", "")
                        
                        # Skip if question is empty
                        if not question:
                            logger.warning(f"Skipping sample {i} from task {task_name}: Empty question")
                            continue
                        
                        # Create record using the proper Argilla Record object
                        record = rg.Record(
                            fields={
                                "question": question,
                                "option_a": option_a,
                                "option_b": option_b,
                                "option_c": option_c,
                                "option_d": option_d,
                                "model_response": model_response,
                                "target_response": target_response
                            }
                        )
                        records.append(record)
                    except Exception as e:
                        logger.error(f"Error processing sample {i} from task {task_name}: {e}")
            
            # Log records to Argilla in batches
            if records:
                logger.info(f"Logging {len(records)} records to Argilla dataset {dataset_name}")
                
                # Process in batches
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    try:
                        # Using dataset.records.log() 
                        dataset.records.log(batch)
                        logger.info(f"Logged batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1} ({len(batch)} records)")
                    except Exception as e:
                        logger.error(f"Error logging batch {i//batch_size + 1}: {e}")
                        logger.error(f"Error details: {str(e)}")
                        
                        # Try again with smaller batch size if batch is too large
                        if batch_size > 1 and len(batch) > 1:
                            smaller_batch_size = max(1, batch_size // 2)
                            logger.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                            for j in range(0, len(batch), smaller_batch_size):
                                smaller_batch = batch[j:j+smaller_batch_size]
                                try:
                                    dataset.records.log(smaller_batch)
                                    logger.info(f"Logged smaller batch ({len(smaller_batch)} records)")
                                except Exception as inner_e:
                                    logger.error(f"Error logging smaller batch: {inner_e}")
                                    
                                    # Try logging records one by one as a last resort
                                    for record in smaller_batch:
                                        try:
                                            dataset.records.log([record])
                                            logger.info("Logged single record")
                                        except Exception as single_e:
                                            logger.error(f"Error logging single record: {single_e}")
                
                logger.info(f"Successfully logged records to Argilla")
                return True
            else:
                logger.warning("No records to log to Argilla")
                return False
        else:
            logger.warning("No samples found in results to log to Argilla")
            return False
            
    except Exception as e:
        logger.error(f"Failed to log results to Argilla: {e}")
        return False

def log_to_mlflow(results):
    """
    Log results to MLflow
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        
        # Function to sanitize metric names for MLflow
        def sanitize_metric_name(name):
            # Replace commas and other invalid characters with underscores
            return re.sub(r'[^a-zA-Z0-9_\-\.\s\:\\/]', '_', name)
        
        # Start a new MLflow run
        model_name = results.get("config", {}).get("model_args", {}).get("model", "unknown")
        with mlflow.start_run(run_name=f"{MODEL_NAME}-{model_name}") as run:
            # Log parameters
            mlflow.log_params({
                "model_type": MODEL_NAME,
                "model_name": model_name,
                "temperature": results.get("config", {}).get("model_args", {}).get("temperature", 0.0),
                "num_fewshot": NUM_FEWSHOT,
                "batch_size": BATCH_SIZE,
                "tasks": "_".join(results.get("n-shot", {}).keys()),
                "system_prompt": SYSTEM_PROMPT if SYSTEM_PROMPT else "Not provided"
            })
            
            # Log metrics
            for task_name, task_results in results.get("results", {}).items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        safe_metric_name = sanitize_metric_name(f"{task_name}_{metric_name}")
                        mlflow.log_metric(safe_metric_name, metric_value)
            
            # Log the results file as an artifact
            results_file = os.path.join(RESULTS_DIR, RESULTS_FILE)
            if os.path.exists(results_file):
                mlflow.log_artifact(results_file)
            
            # Create a visualization of results if possible
            try:
                task_names = []
                exact_match_scores = []
                
                for task_name, task_results in results.get("results", {}).items():
                    if "exact_match" in task_results or "exact_match,none" in task_results:
                        task_names.append(task_name)
                        score = task_results.get("exact_match", task_results.get("exact_match,none", 0))
                        exact_match_scores.append(score)
                
                if task_names and exact_match_scores:
                    plt.figure(figsize=(10, 6))
                    plt.bar(task_names, exact_match_scores)
                    plt.ylim(0, 1.0)
                    plt.ylabel("Exact Match Score")
                    plt.title(f"Model Performance: {model_name}")
                    plot_file = os.path.join(RESULTS_DIR, "performance_plot.png")
                    plt.savefig(plot_file)
                    mlflow.log_artifact(plot_file)
            except Exception as e:
                logger.warning(f"Could not create visualization: {e}")
            
            logger.info(f"Successfully logged results to MLflow. Run ID: {run.info.run_id}")
            return run.info.run_id
    
    except Exception as e:
        logger.error(f"Failed to log results to MLflow: {e}")
        return None

def load_results_file(file_path):
    """
    Load results from a file
    """
    try:
        if not os.path.exists(file_path):
            logger.info(f"Results file not found: {file_path}")
            return None
            
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded results from {file_path}")
        return results
    
    except Exception as e:
        logger.error(f"Error loading results file: {e}")
        return None

def main():
    # Check if results file exists
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    results = load_results_file(results_path)
    
    # Run evaluation if needed
    if results is None:
        logger.info(f"Results file not found or could not be loaded.")
        
        if not EVAL_MODULES_AVAILABLE:
            logger.error("Evaluation modules not available. Cannot run evaluation.")
            sys.exit("Required evaluation modules not found")
        
        logger.info(f"Running evaluation...")
        results = run_evaluation()
        
        if results is None:
            logger.error("Evaluation failed or returned no results.")
            sys.exit("Evaluation failed")
    else:
        logger.info(f"Using existing results from {results_path}")
    
    # Log to MLflow
    logger.info("Logging results to MLflow...")
    mlflow_run_id = log_to_mlflow(results)
    if mlflow_run_id:
        logger.info(f"MLflow run ID: {mlflow_run_id}")
    else:
        logger.warning("Failed to log to MLflow, continuing with Argilla logging")
    
    # Initialize Argilla client and log results
    logger.info("Initializing Argilla client...")
    argilla_client = init_argilla_client()
    
    if argilla_client:
        # Try to create or find dataset
        logger.info(f"Logging results to Argilla dataset: {ARGILLA_DATASET}")
        argilla_success = log_results_to_argilla(
            argilla_client, 
            results,  
            ARGILLA_DATASET, 
            ARGILLA_BATCH_SIZE
        )
        
        if argilla_success:
            logger.info("Successfully logged results to Argilla!")
        else:
            logger.error("Failed to log results to Argilla")
    else:
        logger.error("Could not initialize Argilla client, skipping Argilla logging")
    
    # Print summary
    logger.info("\n===== SUMMARY =====")
    if results and os.path.exists(results_path):
        logger.info(f"Results: Available at {results_path}")
    else:
        logger.info("Results: Not available")
    
    if mlflow_run_id:
        logger.info(f"MLflow: Success (Run ID: {mlflow_run_id})")
    else:
        logger.info("MLflow: Failed or skipped")
    
    if argilla_client and argilla_success:
        logger.info(f"Argilla: Success (Dataset: {ARGILLA_DATASET})")
    else:
        logger.info("Argilla: Failed or skipped")

if __name__ == "__main__":
    main()