import json

def convert_py_to_ipynb(py_file, ipynb_file):
    with open(py_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    current_code = []
    
    for line in lines:
        # Check if line is a major separator
        if line.startswith("# ============================================================================="):
            if current_code:
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_code
                })
                current_code = []
        current_code.append(line)

    if current_code:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": current_code
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(ipynb_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        
if __name__ == "__main__":
    convert_py_to_ipynb("main3.py", "main3.ipynb")
    print("Converted main3.py to main3.ipynb")
