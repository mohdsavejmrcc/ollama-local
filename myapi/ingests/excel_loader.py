import itertools
from pathlib import Path
import pandas as pd
from typing import Any, List, Optional, Union


def load_pandas_excel_data(
    file: Path,
    pandas_config: Optional[dict] = None,
    row_joiner: str = "\n",
    col_joiner: str = " ",
    include_sheetname: bool = False,
    sheet_name: Optional[Union[str, int, list]] = None,
    extra_info: Optional[dict] = None,
) -> List[dict]:
    """Load Excel file using Pandas and return data as a list of dictionaries.

    Args:
        file (Path): Path to the Excel file.
        pandas_config (dict): Options for `pandas.read_excel`.
        row_joiner (str): The separator used to join rows. Default is "\n".
        col_joiner (str): The separator used to join columns. Default is a space (" ").
        include_sheetname (bool): Whether to include the sheet name in the output.
        sheet_name (Union[str, int, list]): Specific sheet(s) to read.
        extra_info (dict): Extra metadata to be added to each document.

    Returns:
        List[dict]: List of dictionaries containing 'text' and 'metadata' for each sheet.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Install pandas using `pip install pandas` to use this function"
        )

    # Set pandas config if provided
    pandas_config = pandas_config or {}

    if sheet_name is not None:
        sheet_name = [sheet_name] if not isinstance(sheet_name, list) else sheet_name

    # Read the Excel file
    dfs = pd.read_excel(file, sheet_name=sheet_name, **pandas_config)
    sheet_names = dfs.keys()
    df_sheets = []

    for key in sheet_names:
        sheet = []
        if include_sheetname:
            sheet.append([key])
        dfs[key] = dfs[key].dropna(axis=0, how="all")
        dfs[key] = dfs[key].fillna("")
        sheet.extend(dfs[key].values.astype(str).tolist())
        df_sheets.append(sheet)

    # Flatten list of lists into a single list
    text_list = list(itertools.chain.from_iterable(df_sheets))

    # Format the output
    output = [
        {
            "text": row_joiner.join(
                col_joiner.join(sublist) for sublist in text_list
            ),
            "metadata": extra_info or {},
        }
    ]
    return output


def load_excel_data(
    file: Path,
    pandas_config: Optional[dict] = None,
    row_joiner: str = "\n",
    col_joiner: str = " ",
    include_sheetname: bool = True,
    sheet_name: Optional[Union[str, int, list]] = None,
    extra_info: Optional[dict] = None,
) -> List[dict]:
    """Load Excel data and return content from multiple sheets.

    Args:
        file (Path): Path to the Excel file.
        pandas_config (dict): Options for `pandas.read_excel`.
        row_joiner (str): The separator used to join rows. Default is "\n".
        col_joiner (str): The separator used to join columns. Default is a space (" ").
        include_sheetname (bool): Whether to include the sheet name in the output.
        sheet_name (Union[str, int, list]): Specific sheet(s) to read.
        extra_info (dict): Extra metadata to be added to each document.

    Returns:
        List[dict]: List of dictionaries containing 'text' and 'metadata' for each sheet.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Install pandas using `pip install pandas` to use this function"
        )

    # Set pandas config if provided
    pandas_config = pandas_config or {}

    if sheet_name is not None:
        sheet_name = [sheet_name] if not isinstance(sheet_name, list) else sheet_name

    # Read the Excel file
    dfs = pd.read_excel(file, sheet_name=sheet_name, **pandas_config)
    sheet_names = dfs.keys()
    output = []

    for idx, key in enumerate(sheet_names):
        dfs[key] = dfs[key].dropna(axis=0, how="all")
        dfs[key] = dfs[key].fillna("")
        rows = dfs[key].values.astype(str).tolist()
        content = row_joiner.join(
            col_joiner.join(row).strip() for row in rows
        ).strip()

        if include_sheetname:
            content = f"(Sheet {key} of file {file.name})\n{content}"

        metadata = {"page_label": idx + 1, "sheet_name": key, **(extra_info or {})}
        output.append({"text": content, "metadata": metadata})

    return output
