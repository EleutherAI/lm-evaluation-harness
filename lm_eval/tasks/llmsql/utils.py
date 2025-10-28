import re
import sqlite3

from huggingface_hub import hf_hub_download


_RE_FROM_TABLE = re.compile(r'FROM\s+"([^"]+)"', re.IGNORECASE)
_RE_SELECT_BLOCK = re.compile(r"(?s)SELECT\b.*?(?=(?:;|\n{4,}|```|$))", re.IGNORECASE)
_RE_SELECT_KEYWORD = re.compile(r"(?i)(SELECT)")


def list_fewshot_samples() -> list[dict]:
    print("list_fewshot_samples used")
    return [
        {
            "question": "What is the price of the Samsung Galaxy S23?",
            "headers": "['Brand', 'Model', 'Price', 'Storage', 'Color']",
            "types": "['text', 'text', 'real', 'text', 'text']",
            "sample_row": "['Apple', 'iPhone 14', 899.99, '128GB', 'White']",
            "sql": """SELECT "Price" FROM "Table" WHERE "Brand" = "Samsung" AND "Model" = "Galaxy S23";""",
            "few_shot": "1",
        },
        {
            "question": "How many books did Maya Chen publish?",
            "headers": "['Author', 'Books Published', 'Genre', 'Country', 'Years Active']",
            "types": "['text', 'real', 'text', 'text', 'text']",
            "sample_row": "['John Smith', 3, 'Non-fiction', 'Canada', '2005–2015']",
            "sql": """SELECT "Books Published" FROM "Table" WHERE "Author" = "Maya Chen";""",
            "few_shot": "1",
        },
        {
            "question": "What is the total population of cities in California?",
            "headers": "['City', 'State', 'Population', 'Area', 'Founded']",
            "types": "['text', 'text', 'real', 'real', 'text']",
            "sample_row": "['Houston', 'Texas', 2304580, 1651.1, '1837']",
            "sql": """SELECT SUM("Population") FROM "Table" WHERE "State" = "California";""",
            "few_shot": "1",
        },
        {
            "question": "How many restaurants serve Italian cuisine?",
            "headers": "['Restaurant', 'Cuisine', 'Rating', 'City', 'Price Range']",
            "types": "['text', 'text', 'real', 'text', 'text']",
            "sample_row": "['Golden Dragon', 'Chinese', 4.2, 'Boston', '$$']",
            "sql": """SELECT COUNT(*) FROM "Table" WHERE "Cuisine" = "Italian";""",
            "few_shot": "1",
        },
        {
            "question": "What is the average salary for Software Engineers?",
            "headers": "['Job Title', 'Salary', 'Experience', 'Location', 'Company Size']",
            "types": "['text', 'real', 'text', 'text', 'text']",
            "sample_row": "['Data Analyst', 70000, 'Junior', 'Chicago', '200–500']",
            "sql": """SELECT AVG("Salary") FROM "Table" WHERE "Job Title" = "Software Engineer";""",
            "few_shot": "1",
        },
    ]


def build_prompt(doc) -> str:
    return f"""### NOW ANSWER:
Question: {doc["question"]}
Columns: {doc["headers"]}
Types: {doc["types"]}
Sample row: {doc["sample_row"]}
SQL:"""


def fix_table_name(sql: str, table_id: str) -> str:
    """
    Replace placeholder table name in the SQL query with the actual table ID.

    During evaluation, the LLM is instructed to always generate queries using
    a generic placeholder table name (`FROM Table`, `FROM "Table"`, or `FROM 'Table'`).
    This keeps the model’s task simpler and avoids requiring it to memorize or
    reproduce arbitrary, dataset-specific table IDs.

    This function post-processes the model’s SQL output by replacing the placeholder
    with the true table identifier for the current question.

    Args:
        sql (str): SQL query string produced by the model, using "Table" as placeholder.
        table_id (str): Actual table name/identifier for the current question.

    Returns:
        str: SQL query with the correct table name substituted.
    """
    return (
        sql.replace("FROM 'Table'", f'FROM "{table_id}"')
        .replace('FROM "Table"', f'FROM "{table_id}"')
        .replace("FROM Table", f'FROM "{table_id}"')
        .strip()
    )


def _download_file(filename: str) -> str:
    file_path = hf_hub_download(
        repo_id="llmsql-bench/llmsql-benchmark-lm-evaluation-harness",
        filename=filename,
        repo_type="dataset",
        local_dir="./llmsql_workdir",
    )
    return file_path


def extract_table_name(sql: str) -> str | None:
    """Extract the name of the table from reference SQL.

    Args:
        sql (str): reference sql.

    Returns:
        str | None: the name of the table.
    """
    match = _RE_FROM_TABLE.search(sql)
    return match.group(1) if match else None


def find_sql(model_output: str, limit: int = 10) -> list[str]:
    """Finds first 10 SQLs from the model output to test.


    Args:
        model_output (str): The unstructured output of the model.
        limit (int): Limit the output number of SQLs. Default 10.

    Returns:
        List[str]:
            - list of SQL queries from the output.
    """
    results = []
    for match in _RE_SELECT_KEYWORD.finditer(model_output):
        start_pos = match.start(1)
        remaining = model_output[start_pos:]
        query_match = _RE_SELECT_BLOCK.search(remaining)
        if query_match:
            query = query_match.group(0).strip()
            if query and query not in results:
                results.append(query)
    return results[:limit]


def execute_sql(cursor: sqlite3.Cursor, sql: str) -> list[tuple] | None:
    """
    Execute a SQL query on the given SQLite connection cursor and return its results.

    The results are always sorted to avoid differences caused by row order (order agnostic).
    If the query fails, the function logs the error and returns None.

    Args:
        conn (sqlite3.Cursor): An active SQLite database cursor.
        sql (str): SQL query string to execute.

    Returns:
        Optional[List[Tuple]]:
            - Sorted list of result rows (each row as a tuple) if successful.
            - [(None,)] if the query executed but returned NULL values.
            - None if the SQL execution failed due to an exception.
    """
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        return sorted(results)
    except Exception:
        return None
