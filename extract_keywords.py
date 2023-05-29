import re

def extract_cpp_keywords(cpp_code):
    # C++ 키워드 리스트
    cpp_keywords = ['while', 'break', 'else' ,'if', 'for', 'int', 'double', 'char', 'new', 'void']

    # C++ 코드에서 주석 제거
    pattern = r'(\/\/.*)|(\".*?\"|\'.*?\')|\/\*[\s\S]*?\*\/'
    cpp_code = re.sub(pattern, '', cpp_code)

    # C++ 코드에서 키워드만 추출
    cpp_tokens = re.findall(r'\b\w+\b', cpp_code)
    cpp_keywords_only = [token for token in cpp_tokens if token in cpp_keywords]

    return cpp_keywords_only