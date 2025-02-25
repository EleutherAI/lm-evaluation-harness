import os
import logging
from dotenv import load_dotenv
from evaluator import AIBEModelEvaluator
from lm_eval.tasks import get_task_dict

def main():
    """
    Main entry point for the AIBE evaluation script
    """
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create evaluator instance
    evaluator_instance = AIBEModelEvaluator(
        model_name="courteasy-ai-gpt4o",  # Use GPT-4
        num_fewshot=5,
        batch_size=4,
        huggingface_token=os.getenv('HUGGINGFACE_TOKEN'),
        argilla_api_key=os.getenv('ARGILLA_API_KEY'),
        argilla_api_url=os.getenv('ARGILLA_API_URL'),
        azure_openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),  # Pass Azure OpenAI API key
        azure_openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')  # Pass Azure OpenAI endpoint
    )

    # Add example output for verification
    print("\nExample prompt format:")
    example_doc = {
        "question": "What is the capital of France?",
        "options": "a) London\nb) Paris\nc) Berlin\nd) Madrid",
        "correct_option": "b"
    }
    task_dict = get_task_dict(["aibe"])
    print(task_dict["aibe"].config["doc_to_text"].format(**example_doc))

    # Run the evaluation
    results = evaluator_instance.run_evaluation()
    
    # Print summary information
    if results and 'results' in results and 'aibe' in results['results']:
        accuracy = results['results']['aibe'].get('acc', None)
        if accuracy is not None:
            print(f"\nEvaluation Results:")
            print(f"Model: {evaluator_instance.model_name}")
            print(f"Accuracy: {accuracy:.2%}")
    
    print("\nEvaluation completed. Check MLflow dashboard and Argilla UI for results.")

if __name__ == "__main__":
    main()