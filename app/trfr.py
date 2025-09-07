import re
import math
import json

# helpers
def latex_to_python_var(name):
    # Remove LaTeX symbols and replace [] with _
    name = name.replace('\\', '')
    name = name.replace('[', '_').replace(']', '')
    name = name.replace('{', '').replace('}', '')
    return name

def replace_var(match):
    var_name = match.group(1)
    py_name = latex_to_python_var(var_name)
    return f"variables['{py_name}']"

# main
def evaluate_trading_formulas(test_cases):
    results = []

    for case in test_cases:
        name = case['name']
        formula = case['formula']
        variables = case['variables']

        print(f"\n=== Test Case: {name} ===")
        print(f"Original formula: {formula}")
        print(f"Variables: {variables}")

        # Remove anything before = (left-hand side)
        if '=' in formula:
            formula_rhs = re.sub(r'\\([A-Za-z0-9_]+)', replace_var, formula_rhs)
        else:
            formula_rhs = formula

        # Replace \text{Var} with variables['Var']
        def replace_var(match):
            var_name = match.group(1)
            return f"variables['{var_name}']"

        formula_rhs = re.sub(r'\\text\{([^\}]+)\}', replace_var, formula_rhs)

        # Replace LaTeX operators with Python equivalents
        formula_rhs = formula_rhs.replace('\\times', '*')
        formula_rhs = formula_rhs.replace('\\cdot', '*')
        formula_rhs = formula_rhs.replace('^', '**')

        # Replace max and min
        formula_rhs = formula_rhs.replace('\\max', 'max')
        formula_rhs = formula_rhs.replace('\\min', 'min')

        # Handle \frac{a}{b} -> (a)/(b)
        def frac_repl(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f"({numerator})/({denominator})"
        formula_rhs = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', frac_repl, formula_rhs)

        # Handle exponentials e^{x} -> math.exp(x)
        formula_rhs = re.sub(r'e\^\{([^\}]+)\}', r'math.exp(\1)', formula_rhs)

        # Remove any remaining $ signs (LaTeX display math)
        formula_rhs = formula_rhs.replace('$', '')

        # Remove spaces to avoid issues
        formula_rhs = formula_rhs.strip()

        print(f"Transformed formula: {formula_rhs}")

        try:
            # Evaluate safely with math functions and max/min
            result = eval(formula_rhs, {"math": math, "variables": variables, "max": max, "min": min})
        except Exception as e:
            print(f"Error evaluating formula: {e}")
            result = None

        results.append({"result": round(result, 4) if result is not None else None})

    return results

# ===== Example usage =====
if __name__ == "__main__":
    test_input = [
        {
          "name": "test1",
          "formula": "Fee = \\text{TradeAmount} \\times \\text{BrokerageRate} + \\text{FixedCharge}",
          "variables": {
            "TradeAmount": 10000.0,
            "BrokerageRate": 0.0025,
            "FixedCharge": 10.0
          },
          "type": "compute"
        },
        {
          "name": "test2",
          "formula": "Fee = \\max(\\text{TradeAmount} \\times \\text{BrokerageRate}, \\text{MinimumFee})",
          "variables": {
            "TradeAmount": 1000.0,
            "BrokerageRate": 0.003,
            "MinimumFee": 15.0
          },
          "type": "compute"
        },
        {
          "name": "test3",
          "formula": "Fee = \\frac{\\text{TradeAmount} - \\text{Discount}}{\\text{ConversionRate}}",
          "variables": {
            "TradeAmount": 11300.0,
            "Discount": 500.0,
            "ConversionRate": 1.2
          },
          "type": "compute"
        },
    ]

    output = evaluate_trading_formulas(test_input)
    print("\n=== Results ===")
    print(json.dumps(output, indent=2))
