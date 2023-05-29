import re
import pprint

def extract_functions(code):
    functions = []
    pattern = re.compile(r'(\b\w+)\s+(\w+)\([^)]*\)\s*{([^}]*)}')
    matches = re.findall(pattern, code)
    for match in matches:
        function = {
            'name': match[1].strip(),
            'definition': match[2].strip()
        }
        functions.append(function)
    return functions

def replace_function_calls(extracted_functions):
    main_function = None
    function_calls = []
    for function in extracted_functions:
        if function['name'] == 'main':
            main_function = function
        else:
            function_calls.append(function['name'])
    
    if main_function is None:
        return extracted_functions
    
    for function_call in function_calls:
        pattern = r'\b' + re.escape(function_call) + r'\([^)]*\)'
        matches = re.findall(pattern, main_function['definition'])
        for match in matches:
            called_function = next((f for f in extracted_functions if f['name'] == function_call), None)
            if called_function is not None:
                main_function['definition'] = main_function['definition'].replace(match, called_function['definition'])
    
    return extracted_functions
