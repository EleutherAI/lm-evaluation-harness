#!/usr/bin/env python3
"""
Demo script showing the Countdown task in action.
"""

import utils

def demo():
    print("=" * 70)
    print("COUNTDOWN TASK DEMO")
    print("=" * 70)
    print()
    
    # Example from the dataset
    doc = {'nums': [41, 70, 18, 35], 'target': 46}
    
    # 1. Show the prompt
    print("1. PROMPT GENERATION")
    print("-" * 70)
    prompt = utils.doc_to_text(doc)
    print(prompt)
    print()
    
    # 2. Show various model responses and their scores
    print("2. MODEL RESPONSES AND SCORING")
    print("-" * 70)
    
    test_responses = [
        ("Correct solution with answer tags", 
         "<answer>(70 - 41) + (35 - 18)</answer>"),
        
        ("Correct solution without tags", 
         "(70 - 41) + (35 - 18)"),
        
        ("Correct solution with explanation", 
         "I can solve this by: (70 - 41) + (35 - 18) = 46"),
        
        ("Alternative correct solution", 
         "70 + 35 - 41 - 18"),
        
        ("Partial solution (not all numbers)", 
         "70 - 41 + 18"),
        
        ("Wrong numbers used", 
         "100 - 50 - 4"),
        
        ("No equation provided", 
         "I cannot solve this puzzle."),
    ]
    
    ground_truth = {'target': 46, 'numbers': [41, 70, 18, 35]}
    
    for description, response in test_responses:
        print(f"\nScenario: {description}")
        print(f"Response: {response}")
        
        # Extract equation
        equation = utils.extract_solution(response)
        print(f"  → Extracted: {equation}")
        
        if equation:
            # Validate
            is_valid = utils.validate_equation(equation, ground_truth['numbers'])
            print(f"  → Valid format: {is_valid}")
            
            # Evaluate
            result = utils.evaluate_equation(equation)
            print(f"  → Evaluates to: {result}")
            
            # Check if correct
            if result is not None:
                is_correct = abs(result - ground_truth['target']) < 1e-5
                print(f"  → Matches target {ground_truth['target']}: {is_correct}")
        
        # Final score
        score = utils.compute_score(response, ground_truth)
        print(f"  ➤ SCORE: {score}")
    
    print()
    print("=" * 70)
    print("SCORING RUBRIC:")
    print("  1.0 = Correct equation (all numbers used, correct result)")
    print("  0.0 = No valid equation found")
    print("=" * 70)

if __name__ == "__main__":
    demo()

