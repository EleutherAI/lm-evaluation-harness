import re
import json
import os
import time
from typing import List, Dict, Any, Optional
import datasets
import torch
from openai import OpenAI
from tqdm import tqdm

# Metric definitions from your original code
METRIC_DEFINITIONS = {
    "query relevance": "How much the information in the response is related to the query, and not necessarily how well it answers the query. A high-score indicates that all the information in the response is relevant to the query. A low-score indicates that the information in the response is largely irrelevant. A score of -1 indicates that this metric is irrelevant for this response.",
    "query completeness": "How thoroughly the response answers the query. A high-score indicates that the response answers all parts of the query. A low-score indicates that the response barely answers the query, or does not answer it at all. A response with highly-relevant information but that does not answer the query at all would get a low score.",
    "context adherence": "How much of the response is directly based on the context. A high-score indicates that the response uses information only from the context, even if it does not answer the query at all. A low score indicates that the response does not use information from the context, even if the response accurately answers the query. However, the response asking for information not found in the context should not be penalized. ",
    "context completeness": "How much out of the total relevant context is incorporated into the response. A high score indicates that the response uses nearly all the relevant information from the context, even if the response quality itself is poor. A low score indicates that the response uses little to none of the relevant information from the context, even if the response is accurate and the quality itself is high.",
    "response coherence": "The overall coherence of the response, such as grammar, fluency and readability. A high score indicates that the response is coherent and easy to read. A low score indicates that the response is incoherent, difficult to read, or contains grammatical errors. This metric should not take into account the content of the response, only the fluency and readability.",
    "response length": "The length of the response, including whether it is too short or too long.",
    "refusal quality": "How well the response explains its refusal to answer the query. A high score indicates that the refusal is well-explained and justified. A low score indicates that the refusal is poorly explained or unjustified. A score of -1 indicates that this metric is irrelevant for this response. This score should be outputted only when the response does not refuse to respond.",
    "refusal clarification quality": "How likely the clarification questions asked during refusal are to lead to a successful response. Use the ground truth answer as a reference when evaluating. For example, if the ground truth answer is provided, evaluate how well the SLM response's clarification questions would likely lead to a similar answer. Similarly, if the ground truth answer contains a refusal, evaluate how well the SLM response's clarification questions align with the ground truth answer's clarifications. A high score indicates that the clarification questions are likely to lead to a successful response OR the questions seek the same information as those of the ground truth answer's questions. A low score indicates that the clarification questions are unlikely to lead to a successful response. A score of -1 indicates that this metric is irrelevant for this response. This score should be outputted only when the response does not refuse to respond.",
    "refusal presence": "A boolean metric to measure if the model response contains a refusal. Do not consider the ground truth answer when evaluating this metric; use only the SLM response. You should ignore the standard 0 to 10 scoring system, and simply output 0 for False (no refusal) and 1 for True (SLM refused to answer). Phrases like 'I don't know', 'I can't answer that', or 'The context does not contain <something>, so I cannot give a complete answer' should be considered refusals. Remember that you are strictly looking at the model response for this metric, and not the context or query. Also remember that if the model response directly cites the context, and the context contains the aforementioned triggering phrases, you should not consider the SLM response a refusal. ",
    "correctness": "How correct the response is. Compare the model's response to the provided ground truth response and assess the model response's overall correctness. A high score indicates that the model response answers the query accurately. Slight rewordings from the provided ground truth answer are permitted, as long as none of the information is conveyed inaccurately. A low score indicates that the model response deviates significantly from the provided ground truth answer in factual accuracy. This metric should not take into account the context, only the correctness of the model's response compared to the ground truth."
}

# Dict to indicate if the given metric (key) should be provided the ground truth answer at judge time.
METRIC_USE_GT = {
    "query relevance": False,
    "query completeness": False,
    "context adherence": False,
    "context completeness": False,
    "response coherence": False,
    "response length": False,
    "refusal quality": False,
    "refusal clarification quality": False,
    "refusal presence": False,
    "correctness": True,
}

def get_metric_definition(metric_name: str) -> str:
    """Get the definition for a specific metric."""
    assert metric_name in METRIC_DEFINITIONS, (
        f"Metric name {metric_name} is not valid. Choose from {list(METRIC_DEFINITIONS.keys())}."
    )
    return METRIC_DEFINITIONS[metric_name]

def metric_needs_gt(metric_name: str) -> bool:
    """Check if a metric requires the ground truth answer for evaluation."""
    assert metric_name in METRIC_USE_GT, (
        f"Metric name {metric_name} is not valid. Choose from {list(METRIC_USE_GT.keys())}."
    )
    return METRIC_USE_GT[metric_name]

def get_available_metrics() -> List[str]:
    """Get list of all available metrics."""
    return list(METRIC_DEFINITIONS.keys())

def format_rag_prompt(query: str, context: str, accepts_sys: bool) -> str:
    """Format RAG prompt for model generation."""
    system_prompt = """
    You are a helpful assistant that provides answers to queries based on the provided context.

    You MUST clearly refuse to answer the query and ask for additional information from the user if the answer cannot be found in the context.
    The output should not contain your judgment on answerability, only your answer OR your refusal + clarifications.

    Stay within the bounds of the provided context and avoid making assumptions.
    """
    user_prompt = f"""
    # Role and Task Description
    Judge if the following query is answerable from ONLY the provided context.
    If so, provide a complete, grounded answer to the query, and do not mention your judgement.
    Try to address all aspects of the query, but if certain parts are not answerable, clearly state that you do not have enough information.

    OTHERWISE, refuse clearly to answer and ask for the additional information you require from the user. 
    You should give a concise explanation of why you cannot answer the query based on the context, and ask for more relevant information from the user.

    # Task
    Given the following query and context, please provide your response:
    Query: {query}

    Context: {context}

    WITHOUT mentioning your judgement either your grounded answer, OR refusal and clarifications:
    """

    messages = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if accepts_sys
        else [{"role": "user", "content": system_prompt + user_prompt}]
    )
    return messages

def _format_judge_system_prompt(metric_name: str, metric_definition: str) -> str:
    """Format system prompt for judge evaluation."""
    assert metric_name in get_available_metrics(), (
        f"Metric name {metric_name} is not valid. Choose from {get_available_metrics()}."
    )

    system_prompt_template = f"""
    # Role and Objective
    You are an expert at evaluating SLM-generated responses according to the following metric: {metric_name.upper()}.
    
    ## Metric Definition
    {metric_name.upper()} is defined as follows: {metric_definition}

    # Evaluation Instructions
    You will be provided a query, a context, and a response. 
    Your task is to evaluate the RESPONSE based on the provided QUERY, CONTEXT, and ONLY if the metric defined above permits, the GROUND TRUTH ANSWER.
    Follow the guidance provided in the metric definition to evaluate the response.

    Be as objective as possible in your evaluation. Do not allow the response's conclusions to influence your own evaluations.

    Output your evaluation according to the JSON format specified.
    """
    return system_prompt_template

def _format_judge_user_prompt(
    metric_name: str, query: str, context: List[str], response: str, ground_truth: str = None
) -> str:
    """Format user prompt for judge evaluation."""
    user_prompt_template = f"""
    # Task Description
    Your job is to evaluate the quality of a SLM-generated response based on the provided metric.
    The metric you are evaluating is: {metric_name.upper()}.

    # Evaluation Outputs
    Your output should be twofold: 1) an analysis, and 2) a numerical score.

    ## Output Directions
    The score should be a number between 0 and 10, following the criteria outlined in the metric definition, unless otherwise defined in the metric definition (e.g. a boolean metric, where you should output 0 for false or 1 for true).

    Complete your concise analysis before outputting your score. The score should be an overall numerical or boolean assessment based on your analysis and the metric definition.

    {'''
    ### Ground Truth Answer
    Certain metrics may require you to compare the SLM response to a ground truth answer.
    The ground truth answer is provided for those metrics ONLY. You should not consider it unless your metric definition instructs you to do so.''' if ground_truth else ''}
    
    # Output Format
    Please provide your evaluation according to the following JSON format:
    {{
        "explanation": "<your analysis>",
        "score": <your score from 0 to 10, or 0/1 if applicable>
    }}
    
    # Do below
    Evaluate the following SLM RESPONSE, given the following QUERY and CONTEXT. Only consider GROUND TRUTH ANSWER if your metric definition instructs you to do so. 
    QUERY: {query}
    CONTEXT: {context}
    SLM RESPONSE: {response}
    {f'GROUND TRUTH ANSWER: {ground_truth}' if ground_truth else ''}
    """
    return user_prompt_template

def format_messages(
    query: str, context: List[str], response: str, metric_name: str, ground_truth: Optional[str] = None
) -> List[Dict[str, str]]:
    """Format the messages for the OpenAI API call."""
    assert metric_name in METRIC_DEFINITIONS, (
        f"Metric name {metric_name} is not valid. Choose from {list(METRIC_DEFINITIONS.keys())}."
    )

    if not metric_needs_gt(metric_name) and ground_truth is not None:
        print(f"Warning: The metric '{metric_name}' does not require a ground truth answer, but one was provided. It will be ignored.")
        ground_truth = None
    elif metric_needs_gt(metric_name) and ground_truth is None:
        print(f"Warning: The metric '{metric_name}' requires a ground truth answer, but none was provided. The evaluation may be inaccurate.")

    metric_definition = METRIC_DEFINITIONS[metric_name]
    system_prompt = _format_judge_system_prompt(metric_name, metric_definition)
    user_prompt = _format_judge_user_prompt(metric_name, query, context, response, ground_truth=ground_truth)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return messages

def format_batch_request_task(
    metric_name: str, eval_datapoint: Dict, generation_args: Dict
):
    """Format a batch request task for the OpenAI API."""
    task = {
        "custom_id": f"{metric_name}_{eval_datapoint['id']}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": generation_args["model"],
            "temperature": generation_args["temperature"],
            "max_tokens": generation_args["max_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": "The score retrieved from the evaluation.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A detailed explanation of the evaluation result.",
                            },
                        },
                        "required": ["score", "explanation"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            "messages": eval_datapoint["messages"],
        },
    }
    return task

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process documents for RAG evaluation."""
    def _process_doc(doc):
        # Handle different dataset formats
        # For RED6k-toy dataset
        if "question" in doc and "contexts" in doc:
            query = doc.get("question", "")
            context = doc.get("contexts", [])
            ground_truth = doc.get("answer", "")
            model_response = doc.get("model_response", "")
        # For other datasets with standard field names
        else:
            query = doc.get("query", "")
            context = doc.get("context", [])
            ground_truth = doc.get("ground_truth", "")
            model_response = doc.get("model_response", "")
        
        # Ensure context is a list of strings
        if isinstance(context, str):
            context = [context]
        elif isinstance(context, list):
            context = [str(ctx) for ctx in context]
        else:
            context = [str(context)]
        
        out_doc = {
            "query": query,
            "context": context,
            "ground_truth": ground_truth,
            "model_response": model_response,
        }
        return out_doc

    return dataset.map(_process_doc)

def doc_to_text(doc) -> str:
    """Format document for text generation."""
    query = doc["query"]
    context = doc["context"]
    
    # Convert context list to string if needed
    if isinstance(context, list):
        context_str = "\n\n".join([f"Document {i+1}: {ctx}" for i, ctx in enumerate(context)])
    else:
        context_str = str(context)
    
    prompt = f"Query: {query}\n\nContext: {context_str}\n\nAnswer:"
    
    return prompt

def doc_to_target(doc) -> str:
    """Get target response for evaluation."""
    return doc.get("ground_truth", "")

def evaluate_with_gpt4(
    query: str, 
    context: List[str], 
    response: str, 
    metric_name: str, 
    ground_truth: str = None,
    openai_client: OpenAI = None,
    gpt_model: str = "gpt-4o"
) -> Dict[str, Any]:
    """Evaluate a single response using GPT-4."""
    if openai_client is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        openai_client = OpenAI(api_key=openai_api_key)
    
    messages = format_messages(query, context, response, metric_name, ground_truth)
    
    generation_args = {
        "model": gpt_model,
        "temperature": 0.3,
        "max_tokens": 512,
        "frequency_penalty": 0.5,
    }
    
    task = format_batch_request_task(metric_name, {"id": "single_eval", "messages": messages}, generation_args)
    
    print(f"Making OpenAI request for metric: {metric_name}")
    print(f"Model: {gpt_model}")
    print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    try:
        response = openai_client.chat.completions.create(
            model=task["body"]["model"],
            messages=task["body"]["messages"],
            temperature=task["body"]["temperature"],
            max_tokens=task["body"]["max_tokens"],
            response_format=task["body"]["response_format"],
        )
        
        print(f"OpenAI request successful for metric: {metric_name}")
        result = json.loads(response.choices[0].message.content)
        print(f"Evaluation result: score={result.get('score', 'N/A')}")
        return result
    except Exception as e:
        print(f"OpenAI request failed for metric: {metric_name}")
        print(f"Error: {e}")
        return {"score": 0.0, "explanation": f"Evaluation failed: {str(e)}"}

def aggregate_rag_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate RAG evaluation metrics."""
    if not results:
        return {}
    
    # Extract scores from results
    scores = []
    explanations = []
    
    for result in results:
        if isinstance(result, dict):
            # Handle different possible score keys
            score = None
            for key in ["relevance_score", "score", "rag_score"]:
                if key in result and isinstance(result[key], (int, float)):
                    score = float(result[key])
                    break
            
            if score is not None:
                scores.append(score)
                explanations.append(result.get("explanation", ""))
            else:
                print(f"Warning: No valid score found in result: {result}")
                scores.append(0.0)
        elif isinstance(result, (int, float)):
            scores.append(float(result))
        else:
            print(f"Warning: Unexpected result format: {result}")
            scores.append(0.0)
    
    if not scores:
        return {}
    
    # Calculate statistics
    mean_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    return {
        "mean_score": mean_score,
        "min_score": min_score,
        "max_score": max_score,
        "num_evaluations": len(scores)
    }

def process_results(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process evaluation results for RAG task."""
    # This function will be called by the framework to process results
    # For RAG evaluation, we need to evaluate the model's response using GPT-4
    query = doc["query"]
    context = doc["context"]
    model_response = doc.get("model_response", "")
    ground_truth = doc.get("ground_truth", "")
    
    print(f"Processing RAG evaluation for query: {query[:50]}{'...' if len(query) > 50 else ''}")
    
    # Get the metric to evaluate (this would be configured in the YAML)
    # For now, we'll use a default metric
    metric_name = "correctness"  # This should be configurable
    
    # Evaluate with GPT-4
    evaluation_result = evaluate_with_gpt4(
        query=query,
        context=context,
        response=model_response,
        metric_name=metric_name,
        ground_truth=ground_truth
    )
    
    print(f"RAG evaluation completed. Score: {evaluation_result.get('score', 'N/A')}")
    
    return {
        "rag_score": evaluation_result["score"],
        "rag_explanation": evaluation_result["explanation"],
        "query": query,
        "context": context,
        "model_response": model_response,
        "ground_truth": ground_truth,
        "metric": metric_name
    }

def process_results_relevance(doc: Dict, results: List[str]) -> Dict[str, Any]:
    """Process evaluation results specifically for query relevance metric."""
    query = doc["query"]
    context = doc["context"]
    ground_truth = doc.get("ground_truth", "")
    
    print(f"Processing relevance evaluation for query: {query[:50]}{'...' if len(query) > 50 else ''}")
    
    # Use the model response from the framework results instead of doc
    if results and len(results) > 0:
        model_response = results[0]
    else:
        model_response = doc.get("model_response", "")
    
    # Evaluate with GPT-4 for query relevance metric
    metric_name = "query relevance"
    evaluation_result = evaluate_with_gpt4(
        query=query,
        context=context,
        response=model_response,
        metric_name=metric_name,
        ground_truth=ground_truth
    )
    
    print(f"Relevance evaluation completed. Score: {evaluation_result.get('score', 'N/A')}")
    return {"relevance_score": evaluation_result.get("score", 0.0)} 