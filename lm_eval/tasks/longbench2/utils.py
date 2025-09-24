def filter_agent_history(resps, docs):
    """Filter function for Long-dialogue History Understanding - Agent history QA"""
    filtered_resps = []
    for resp, doc in zip(resps, docs):
        if doc['domain'] == 'Long-dialogue History Understanding' and doc['sub_domain'] == 'Agent history QA':
            filtered_resps.append(resp)
    return filtered_resps


def process_docs_agent_history(dataset):
    """Process docs to filter for Long-dialogue History Understanding - Agent history QA"""
    return dataset.filter(lambda x: x['domain'] == 'Long-dialogue History Understanding' and x['sub_domain'] == 'Agent history QA')


def process_docs_dialogue_history(dataset):
    """Process docs to filter for Long-dialogue History Understanding - Dialogue history QA"""
    return dataset.filter(lambda x: x['domain'] == 'Long-dialogue History Understanding' and x['sub_domain'] == 'Dialogue history QA')