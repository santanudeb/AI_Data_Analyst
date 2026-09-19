import pandas as pd
import duckdb
from langchain_ollama import ChatOllama

# Local LLM used to convert natural-language questions into SQL.
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

def load_dataset(uploaded_file):
    """Load an uploaded CSV or Excel file into a pandas DataFrame."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV, XLSX, and XLS files are supported.")

    return df

def _schema_text(df):
    """Create a compact schema description for the LLM."""
    lines = []

    for column in df.columns:
        dtype = str(df[column].dtype)

        sample_values = (
            df[column]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        lines.append(
            f"- {column}: {dtype}; examples: {sample_values}"
        )

    return "\n".join(lines)

def _clean_sql(sql):
    """Remove markdown code fences from an LLM-generated SQL query."""
    sql = sql.strip()

    if sql.startswith("```"):
        sql = sql.replace("```sql", "", 1)
        sql = sql.replace("```SQL", "", 1)
        sql = sql.replace("```", "")

    return sql.strip().rstrip(";")

def ask_dataset_question(df, question):
    """
    Answer a natural-language question about the uploaded dataset.

    Flow:
    User question -> Llama 3.2 -> SQL -> DuckDB -> result -> Llama 3.2
    """

    if df is None or df.empty:
        return {
            "answer": "The uploaded dataset is empty.",
            "sql": None
        }

    schema = _schema_text(df)

    sql_prompt = f"""
You are a data analyst.

Convert the user's question into ONE DuckDB SQL query.

The pandas DataFrame is available as a DuckDB table named dataset.

Dataset:
- Rows: {len(df)}
- Columns: {len(df.columns)}

Schema:
{schema}

Rules:
1. Return ONLY SQL. No explanation.
2. Use only the table dataset.
3. Use only columns that exist in the schema.
4. For text comparisons, use case-insensitive matching when appropriate.
5. Use standard DuckDB SQL.
6. Do not INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH, or read files.
7. If the question asks for a count, aggregation, average, minimum, maximum,
   grouping, filtering, sorting, or comparison, write the SQL needed to answer it.
8. Limit detailed row results to 20 rows unless the question explicitly asks
   for a larger number.
9. Do not invent columns.

User question:
{question}
"""
    sql_response = llm.invoke(sql_prompt)
    sql = _clean_sql(sql_response.content)

    # Basic protection against destructive/file-access SQL.
    forbidden = [
        "insert ", "update ", "delete ", "drop ", "alter ",
        "create ", "copy ", "attach ", "install ", "load ",
        "read_csv", "read_parquet", "httpfs"
    ]

    sql_lower = sql.lower()

    if any(word in sql_lower for word in forbidden):
        return {
            "answer": "I could not safely execute the generated data query.",
            "sql": sql
        }

    try:
        # Register the uploaded DataFrame using the same table name used by the LLM-generated SQL.
        con = duckdb.connect()
        con.register("dataset", df)

        # Execute the generated SQL against the uploaded DataFrame.
        result_df = con.execute(sql).df()

        con.close()

    except Exception as exc:
        return {
            "answer": (
                "I couldn't execute the generated query. "
                f"Please try rephrasing the question.\n\nError: {exc}"
            ),
            "sql": sql
        }

    # Convert the query result to compact text for the local LLM.
    if result_df.empty:
        result_text = "The query returned no rows."
    else:
        result_text = result_df.head(20).to_string(index=False)

    answer_prompt = f"""
You are an AI data analyst.

Answer the user's question using ONLY the query result below.

User question:
{question}

Query result:
{result_text}

Rules:
- Give a clear, concise answer.
- Include important numbers.
- If the result contains grouped rows, summarize the relevant groups.
- Do not invent facts that are not present in the query result.
- Do not mention SQL unless it helps explain the answer.
"""
    answer_response = llm.invoke(answer_prompt)

    return {
        "answer": answer_response.content,
        "sql": sql
    }