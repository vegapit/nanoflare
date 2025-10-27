import xml.etree.ElementTree as ET
import argparse, sys
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Catch2 XML Structure Keys
# We are looking for the <BenchmarkResults> tag, which is nested inside the test structure.
BENCHMARK_TAG = 'BenchmarkResults'
MEAN_TIME_TAG = 'mean'
STD_DEV_TAG = 'standardDeviation'
NAME_ATTR = 'name'
VALUE_ATTR = 'value'
UNIT_ATTR = 'unit'

def parse_catch2_xml(xml_file_path: str) -> List[Dict[str, Any]]:
    """
    Parses the Catch2 XML output file to extract benchmark results.

    Args:
        xml_file_path: Path to the Catch2 XML file.

    Returns:
        A list of dictionaries, where each dictionary represents a benchmark result.
    """
    results = []

    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: File not found at '{xml_file_path}'", file=sys.stderr)
        return []
    except ET.ParseError:
        print(f"Error: Failed to parse XML file at '{xml_file_path}'. Ensure it's valid Catch2 XML.", file=sys.stderr)
        return []

    # Iterate through all descendant elements to find BenchmarkResults
    for element in root.iter(BENCHMARK_TAG):
        benchmark_name = element.attrib.get(NAME_ATTR, 'Unnamed Benchmark')

        mean_value: Optional[float] = None
        mean_unit: Optional[str] = None
        stddev_value: Optional[float] = None

        # Extract Mean and Standard Deviation from nested tags
        for child in element:
            if child.tag == MEAN_TIME_TAG:
                mean_value = float(child.attrib.get(VALUE_ATTR, 0.0))
                mean_unit = child.attrib.get(UNIT_ATTR, 'ms')
            elif child.tag == STD_DEV_TAG:
                stddev_value = float(child.attrib.get(VALUE_ATTR, 0.0))

        if mean_value is not None:
            results.append({
                'name': benchmark_name,
                'mean_value': mean_value,
                'mean_unit': mean_unit,
                'stddev_value': stddev_value
            })
    return results

def format_as_markdown_table(benchmark_data: List[Dict[str, Any]]) -> str:
    """
    Formats the extracted benchmark data into a Markdown table string.
    """
    if not benchmark_data:
        return "No benchmark results found in the XML data."

    # Define headers and column names for easy iteration
    headers = ["Benchmark Name", "Mean Time", "Std Dev"]
    keys = ['name', 'mean_value', 'stddev_value']

    # Convert data to string format for length calculation
    str_data = []
    for row in benchmark_data:
        mean_str = f"{1e-6 * row['mean_value']:.2f} {row['mean_unit']}"
        stddev_str = f"{1e-6 * row['stddev_value']:.2f} {row['mean_unit']}"
        str_data.append([row['name'], mean_str, stddev_str])

    # Calculate max width for each column
    col_widths = [len(header) for header in headers]
    for row in str_data:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(item))

    # --- Build the table ---
    
    table_lines = []

    # 1. Header Row
    header_line = "| " + " | ".join(
        header.ljust(col_widths[i]) for i, header in enumerate(headers)
    ) + " |"
    table_lines.append(header_line)

    # 2. Separator Row
    separator_line = "|-" + "-|-".join(
        "-" * col_widths[i] for i in range(len(headers))
    ) + "-|"
    table_lines.append(separator_line)

    # 3. Data Rows
    for row in str_data:
        data_line = "| " + " | ".join(
            row[i].ljust(col_widths[i]) for i in range(len(headers))
        ) + " |"
        table_lines.append(data_line)

    return "\n".join(table_lines)


def main():
    """Main function to handle command-line arguments and run the parsing."""
    parser = argparse.ArgumentParser(
        description="Parse Catch2 XML benchmark output and display results in a single Markdown table."
    )
    parser.add_argument(
        "xml_file",
        help="Path to the Catch2 XML output file (generated with --reporter xml)."
    )

    args = parser.parse_args()

    benchmark_data = parse_catch2_xml(args.xml_file)
    
    # Sort the benchmark data alphabetically by name before formatting
    benchmark_data.sort(key=lambda x: x['name'])

    markdown_table = format_as_markdown_table(benchmark_data)
    
    if benchmark_data:
        print("\n--- Consolidated Benchmark Results ---\n")
    print(markdown_table)

if __name__ == "__main__":
    main()