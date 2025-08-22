"""
Utility functions for babilong benchmark.
"""

import datasets
from datasets import concatenate_datasets


# All available length splits
LENGTH_SPLITS = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M']


def doc_to_text(doc):
    """
    Convert document to input text format.
    
    Args:
        doc: Document containing 'input' and 'question' fields
        
    Returns:
        str: Formatted text for model input
    """
    return f"{doc['input']}\n\nQ: {doc['question']}\nA:"


def doc_to_target(doc):
    """
    Extract target answer from document.
    
    Args:
        doc: Document containing 'target' field
        
    Returns:
        str: Target answer
    """
    return doc['target']


# QA-specific doc_to_text functions based on DEFAULT_PROMPTS
def doc_to_text_qa1(doc):
    """QA1: Single supporting fact - Where is person?"""
    instruction = ('I will give you context with the facts about positions of different persons hidden in some random text '
                  'and a question. You need to answer the question based only on the information from the facts. '
                  'If a person was in different locations, use the latest location to answer the question.')
    
    examples = ('<example>\n'
               'Charlie went to the hallway. Judith come back to the kitchen. Charlie travelled to balcony. '
               'Where is Charlie?\n'
               'Answer: balcony.\n'
               '</example>\n\n'
               '<example>\n'
               'Alan moved to the garage. Charlie went to the beach. Alan went to the shop. Rouse '
               'travelled to balcony. Where is Alan?\n'
               'Answer: shop.\n'
               '</example>')
    
    post_prompt = 'Your answer should contain only one word - location. Do not write anything else after that.'
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa2(doc):
    """QA2: Two supporting facts - Where is object?"""
    instruction = ('I give you context with the facts about locations and actions of different persons '
                  'hidden in some random text and a question.'
                  'You need to answer the question based only on the information from the facts.\n'
                  'If a person got an item in the first location and travelled to the second location '
                  'the item is also in the second location. '
                  'If a person dropped an item in the first location and moved to the second location '
                  'the item remains in the first location.')
    
    examples = ('<example>\n'
               'Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the balcony. '
               'Where is the bottle?\n'
               'Answer: balcony.\n'
               '</example>\n'
               '<example>\n'
               'Alan moved to the garage. Alan got a screw driver. Alan moved to the kitchen. Where '
               'is the screw driver?\n'
               'Answer: kitchen.\n'
               '</example>')
    
    post_prompt = 'Your answer should contain only one word - location. Do not write anything else after that.'
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa3(doc):
    """QA3: Three supporting facts - Where was object before location?"""
    instruction = ('I give you context with the facts about locations and actions of different persons '
                  'hidden in some random text and a question. '
                  'You need to answer the question based only on the information from the facts.\n'
                  'If a person got an item in the first location and travelled to the second location '
                  'the item is also in the second location. '
                  'If a person dropped an item in the first location and moved to the second location '
                  'the item remains in the first location.')
    
    examples = ('<example>\n'
               'John journeyed to the bedroom. Mary grabbed the apple. Mary went back to the bathroom. '
               'Daniel journeyed to the bedroom. Daniel moved to the garden. Mary travelled to the kitchen. '
               'Where was the apple before the kitchen?\n'
               'Answer: bathroom.\n'
               '</example>\n'
               '<example>\n'
               'John went back to the bedroom. John went back to the garden. John went back to the kitchen. '
               'Sandra took the football. Sandra travelled to the garden. Sandra journeyed to the bedroom. '
               'Where was the football before the bedroom?\n'
               'Answer: garden.\n'
               '</example>')
    
    post_prompt = 'Your answer should contain only one word - location. Do not write anything else after that.'
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa4(doc):
    """QA4: Two arg relations - Spatial reasoning"""
    instruction = ('I will give you context with the facts about different people, their location and actions, hidden in '
                  'some random text and a question. '
                  'You need to answer the question based only on the information from the facts.')
    
    examples = ('<example>\n'
               'The hallway is south of the kitchen. The bedroom is north of the kitchen. '
               'What is the kitchen south of?\n'
               'Answer: bedroom\n'
               '</example>\n'
               '<example>\n'
               'The garden is west of the bedroom. The bedroom is west of the kitchen. What is west of the bedroom?\n'
               'Answer: garden\n'
               '</example>')
    
    post_prompt = 'Your answer should contain only one word - location. Do not write anything else after that.'
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa5(doc):
    """QA5: Three arg relations - Who gave what to whom?"""
    instruction = ('I will give you context with the facts about locations and their relations hidden in some random text '
                  'and a question. You need to answer the question based only on the information from the facts.')
    
    examples = ('<example>\n'
               'Mary picked up the apple there. Mary gave the apple to Fred. Mary moved to the bedroom. '
               'Bill took the milk there. Who did Mary give the apple to?\n'
               'Answer: Fred\n'
               '</example>\n'
               '<example>\n'
               'Jeff took the football there. Jeff passed the football to Fred. Jeff got the milk there. '
               'Bill travelled to the bedroom. Who gave the football?\n'
               'Answer: Jeff\n'
               '</example>\n'
               '<example>\n'
               'Fred picked up the apple there. Fred handed the apple to Bill. Bill journeyed to the bedroom. '
               'Jeff went back to the garden. What did Fred give to Bill?\n'
               'Answer: apple\n'
               '</example>')
    
    post_prompt = ('Your answer should contain only one word. Do not write anything else after that. '
                  'Do not explain your answer.')
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa6(doc):
    """QA6: Yes/No questions - Is person in location?"""
    instruction = ('I will give you context with the facts about people and their locations hidden in some random text and a '
                  'question. You need to answer the question based only on the information from the facts. '
                  'If a person was in different locations, use the latest location the person was in to answer the question.')
    
    examples = ('<example>\n'
               'John travelled to the hallway. John travelled to the garden. Is John in the garden?\n'
               'Answer: yes\n'
               '</example>\n'
               '<example>\n'
               'Mary went to the office. Daniel journeyed to the hallway. Mary went to the bedroom. '
               'Sandra went to the garden. Is Mary in the office?\n'
               'Answer: no\n'
               '</example>\n')
    
    post_prompt = ('Your answer should contain only one word - $yes$ or $no$. Do not write anything else after that. '
                  'Do not explain your answer.')
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa7(doc):
    """QA7: Counting - How many objects is person carrying?"""
    instruction = ('I will give you context with the facts about people and objects they carry, hidden in some random text '
                  'and a question. You need to answer the question based only on the information from the facts.')
    
    examples = ('<example>\n'
               'Daniel went to the bedroom. Daniel got the apple there. How many objects is Daniel carrying?\n'
               'Answer: one\n'
               '</example>\n'
               '<example>\n'
               'Mary grabbed the apple there. Mary gave the apple to John. How many objects is Mary carrying?\n'
               'Answer: none\n'
               '</example>\n'
               '<example>\n'
               'Sandra travelled to the hallway. Sandra picked up the milk there. Sandra took the apple there. '
               'Mary travelled to the garden. How many objects is Sandra carrying?\n'
               'Answer: two\n'
               '</example>\n')
    
    post_prompt = ('Your answer should contain only one word - $none$ or $number_of_objects$. '
                  'Do not write anything else after that. Do not explain your answer.')
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa8(doc):
    """QA8: Lists/Sets - What is person carrying?"""
    instruction = ('I will give you context with the facts about people and objects they carry, hidden in some random text '
                  'and a question. You need to answer the question based only on the information from the facts.')
    
    examples = ('<example>\n'
               'Sandra travelled to the garden. Mary grabbed the milk there. What is Mary carrying?\n'
               'Answer: milk\n'
               '</example>\n'
               '<example>\n'
               'Mary travelled to the kitchen. Sandra travelled to the office. John travelled to the office. '
               'Sandra discarded the milk there. What is Sandra carrying?\n'
               'Answer: nothing\n'
               '</example>\n'
               '<example>\n'
               'Daniel grabbed the apple there. Mary went to the office. Daniel moved to the garden. '
               'Daniel grabbed the milk there. Mary went to the kitchen. What is Daniel carrying?\n'
               'Answer: apple,milk\n'
               '</example>\n')
    
    post_prompt = ('Your answer should contain only one or two words: $nothing$ or $object$ or $object_1$, $object_2$. '
                  'Do not write anything else. Do not explain your answer.')
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa9(doc):
    """QA9: Simple negation - Negative statements"""
    instruction = ('I will give you context with the facts about people and their locations hidden in some random text and '
                  'a question. You need to answer the question based only on the information from the facts. '
                  'If a person was in different locations, use the latest location the person was in to answer the question.')
    
    examples = ('<example>\n'
               'John is not in the bathroom. Sandra is not in the bedroom. Is John in the bathroom?\n'
               'Answer: no\n'
               '</example>\n'
               '<example>\n'
               'Mary journeyed to the kitchen. John is in the bedroom. Sandra is not in the garden. '
               'Is Mary in the kitchen?\n'
               'Answer: yes\n'
               '</example>\n')
    
    post_prompt = ('Your answer should contain only one word - $yes$ or $no$. Do not write anything else. '
                  'Do not explain your answer.')
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def doc_to_text_qa10(doc):
    """QA10: Indefinite knowledge - Maybe answers"""
    instruction = ('I will give you context with the facts about people and their locations hidden in some random text and a '
                  'question. You need to answer the question based only on the information from the facts. '
                  'If a person was in different locations, use the latest location the person was in to answer the question.')
    
    examples = ('<example>\n'
               'Bill is in the kitchen. Julie is either in the school or the cinema. Is Bill in the bedroom?\n'
               'Answer: no\n'
               '</example>\n'
               '<example>\n'
               'Fred is in the bedroom. Mary is either in the school or the cinema. Is Mary in the school?\n'
               'Answer: maybe\n'
               '</example>\n'
               '<example>\n'
               'Fred is either in the kitchen or the park. Bill moved to the cinema. Is Bill in the cinema?\n'
               'Answer: yes\n'
               '</example>\n')
    
    post_prompt = ('Your answer should contain only one word - $yes$ or $no$ or $maybe$. Do not write anything else. '
                  'Do not explain your answer.')
    
    return f"{instruction}\n\n{examples}\n\n{doc['input']}\n\n{doc['question']}\n\n{post_prompt}\n\nAnswer:"


def load_babilong_all_lengths(qa_config):
    """
    Load all length splits for a given QA config and concatenate them.
    
    Args:
        qa_config: QA task name (e.g., 'qa1', 'qa2', etc.)
        
    Returns:
        dict: Dictionary with 'test' split containing all length data
    """
    try:
        from datasets import load_dataset
        
        # Load all length splits for the given QA config
        all_datasets = []
        for length_split in LENGTH_SPLITS:
            try:
                dataset = load_dataset('RMT-team/babilong', qa_config, split=length_split)
                # Add length information to each example
                dataset = dataset.map(lambda x: {**x, 'length': length_split})
                all_datasets.append(dataset)
            except Exception as e:
                print(f"Warning: Could not load {qa_config} {length_split}: {e}")
                continue
        
        # Concatenate all datasets
        if all_datasets:
            combined_dataset = concatenate_datasets(all_datasets)
            return {"test": combined_dataset}
        else:
            raise ValueError(f"No valid datasets found for {qa_config}")
            
    except Exception as e:
        print(f"Error loading babilong data for {qa_config}: {e}")
        # Fallback to default loading
        return None