
import re

def roman_to_int(s):
    """Convert Roman numeral to integer"""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_value = 0
    
    for char in reversed(s):
        value = roman_map[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total

def english_to_int(s):
    """Convert English number words to integer"""
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000
    }
    
    # Handle hyphenated numbers and normalize
    words = s.lower().replace('-', ' ').replace(',', '').split()
    total = 0
    current = 0
    
    for word in words:
        if word in word_to_num:
            value = word_to_num[word]
            if value == 100:
                if current == 0:
                    current = 1
                current *= value
            elif value >= 1000:
                if current == 0:
                    current = 1
                total += current * value
                current = 0
            else:
                current += value
    
    return total + current

def detect_language_and_convert(s):
    """Detect the language of a number string and convert to integer with priority"""
    # Check if it's a Roman numeral (priority 0)
    if re.match(r'^[IVXLCDM]+$', s.upper()):
        return roman_to_int(s.upper()), 0
    
    # Check if it's an Arabic numeral (priority 5)
    if re.match(r'^\d+$', s):
        return int(s), 5
    
    # Check if it contains Chinese characters (priority 2 for Traditional, 3 for Simplified)
    chinese_chars = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                    '百', '千', '万', '萬', '億', '亿']
    
    if any(char in s for char in chinese_chars):
        # Check for traditional characters
        if any(char in s for char in ['萬', '億']):
            return chinese_to_int(s), 2  # Traditional Chinese
        else:
            return chinese_to_int(s), 3  # Simplified Chinese
    
    # Check if it's English (priority 1)
    english_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
                    'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
                    'hundred', 'thousand', 'million', 'billion']
    
    s_lower = s.lower()
    if any(word in s_lower for word in english_words):
        return english_to_int(s), 1  # English
    
    # Check if it's German (priority 4)
    german_words = ['null', 'eins', 'ein', 'eine', 'zwei', 'zwo', 'drei', 'vier', 'fünf', 
                   'sechs', 'sieben', 'acht', 'neun', 'zehn', 'elf', 'zwölf', 'dreizehn', 
                   'vierzehn', 'fünfzehn', 'sechzehn', 'siebzehn', 'achtzehn', 'neunzehn', 
                   'zwanzig', 'dreißig', 'vierzig', 'fünfzig', 'sechzig', 'siebzig', 
                   'achtzig', 'neunzig', 'hundert', 'tausend', 'million', 'millionen', 'und']
    
    if any(word in s_lower for word in german_words):
        return german_to_int(s), 4  # German
    
    # Default case - try to parse as integer
    try:
        return int(s), 5
    except ValueError:
        return 0, 5

def duolingo_sort(data):
    """Main function to handle duolingo sort requests"""
    try:
        part = data.get('part', '')
        unsorted_list = data.get('challengeInput', {}).get('unsortedList', [])
        
        if part == 'ONE':
            # Part 1: Only Roman and Arabic numerals, return as integers
            converted = []
            for item in unsorted_list:
                if re.match(r'^[IVXLCDM]+$', item.upper()):
                    converted.append((roman_to_int(item.upper()), item))
                else:
                    converted.append((int(item), item))
            
            # Sort by numerical value
            converted.sort(key=lambda x: x[0])
            # Return as string representations of integers
            sorted_list = [str(x[0]) for x in converted]
            
        elif part == 'TWO':
            # Part 2: Multiple languages, maintain original representation
            converted = []
            for item in unsorted_list:
                value, priority = detect_language_and_convert(item)
                converted.append((value, priority, item))
            
            # Sort by value first, then by language priority for ties
            converted.sort(key=lambda x: (x[0], x[1]))
            # Return original representations
            sorted_list = [x[2] for x in converted]
            
        else:
            return {'error': 'Invalid part specified'}, 400
        
        return {'sortedList': sorted_list}
    
    except Exception as e:
        return {'error': str(e)}, 500

# Test cases for validation
def test_duolingo_examples():
    """Test with the examples from the problem statement"""
    
    # Test Part 1
    part1_request = {
        "part": "ONE",
        "challenge": 1000,
        "challengeInput": {
            "unsortedList": ["MCMXCIX", "C", "XL", "XLIX", "X", "2000"]
        }
    }
    
    result1 = duolingo_sort(part1_request)
    print("Part 1 Result:", result1)
    expected1 = ["10", "40", "49", "100", "1999", "2000"]
    print("Expected:", expected1)
    print("Match:", result1['sortedList'] == expected1)
    print()
    
    # Test Part 2
    part2_request = {
        "part": "TWO",
        "challenge": 2000,
        "challengeInput": {
            "unsortedList": ["MCMXCIX", "one hundred twenty", "四十五", "XLIX", "五萬四千三百二十一", "siebenundachtzig", "dreihundertelf", "one thousand one hundred", "100000"]
        }
    }
    
    result2 = duolingo_sort(part2_request)
    print("Part 2 Result:", result2)
    expected2 = ["四十五", "XLIX", "siebenundachtzig", "one hundred twenty", "dreihundertelf", "one thousand one hundred", "MCMXCIX", "五萬四千三百二十一", "100000"]
    print("Expected:", expected2)
    print("Match:", result2['sortedList'] == expected2)

if __name__ == "__main__":
    test_duolingo_examples()

def chinese_to_int(s):
    """Convert Chinese number characters to integer"""
    # Mapping for Chinese digits and multipliers
    char_to_num = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000, '万': 10000, '萬': 10000,
        '億': 100000000, '亿': 100000000
    }
    
    result = 0
    current_section = 0
    current_number = 0
    has_digit = False
    
    for char in s:
        if char in char_to_num:
            value = char_to_num[char]
            
            if value >= 0 and value <= 9:  # Digit
                current_number = value
                has_digit = True
            elif value == 10:  # 十
                if current_number == 0 and not has_digit:
                    current_number = 1
                current_section += current_number * value
                current_number = 0
                has_digit = False
            elif value == 100:  # 百
                if current_number == 0:
                    current_number = 1
                current_section += current_number * value
                current_number = 0
                has_digit = False
            elif value == 1000:  # 千
                if current_number == 0:
                    current_number = 1
                current_section += current_number * value
                current_number = 0
                has_digit = False
            elif value == 10000:  # 万/萬
                if current_section == 0:
                    current_section = current_number if current_number > 0 else 1
                else:
                    current_section += current_number
                result += current_section * value
                current_section = 0
                current_number = 0
                has_digit = False
            elif value == 100000000:  # 億/亿
                if current_section == 0:
                    current_section = current_number if current_number > 0 else 1
                else:
                    current_section += current_number
                result += current_section * value
                current_section = 0
                current_number = 0
                has_digit = False
    
    # Add remaining numbers
    current_section += current_number
    result += current_section
    
    return result

def german_to_int(s):
    """Convert German number words to integer"""
    # Comprehensive German number mapping
    word_to_num = {
        'null': 0, 'eins': 1, 'ein': 1, 'eine': 1, 'zwei': 2, 'zwo': 2, 'drei': 3, 
        'vier': 4, 'fünf': 5, 'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
        'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
        'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19,
        'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50, 'sechzig': 60,
        'siebzig': 70, 'achtzig': 80, 'neunzig': 90,
        'hundert': 100, 'tausend': 1000, 'million': 1000000, 'millionen': 1000000
    }
    
    # Generate compound numbers for German
    expanded_map = dict(word_to_num)
    
    # Add numbers like einundzwanzig, zweiundzwanzig, etc.
    ones = ['ein', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun']
    tens = [(20, 'zwanzig'), (30, 'dreißig'), (40, 'vierzig'), (50, 'fünfzig'), 
            (60, 'sechzig'), (70, 'siebzig'), (80, 'achtzig'), (90, 'neunzig')]
    
    for i, one in enumerate(ones, 1):
        for ten_val, ten_word in tens:
            compound = f"{one}und{ten_word}"
            expanded_map[compound] = i + ten_val
    
    # Add hundreds
    for i, one in enumerate(['ein', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun'], 1):
        expanded_map[f"{one}hundert"] = i * 100
    
    # Direct lookup for common patterns
    s_lower = s.lower()
    if s_lower in expanded_map:
        return expanded_map[s_lower]
    
    # Complex parsing for multi-part German numbers
    parts = re.split(r'(hundert|tausend|million|millionen)', s_lower)
    total = 0
    current = 0
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part in expanded_map:
            value = expanded_map[part]
            if part in ['hundert', 'tausend', 'million', 'millionen']:
                if current == 0:
                    current = 1
                if part == 'hundert':
                    current *= 100
                elif part == 'tausend':
                    total += current * 1000
                    current = 0
                elif part in ['million', 'millionen']:
                    total += current * 1000000
                    current = 0
            else:
                current += value
        else:
            # Try to parse compound numbers within the part
            if 'und' in part:
                und_parts = part.split('und')
                if len(und_parts) == 2 and und_parts[0] in expanded_map and und_parts[1] in expanded_map:
                    current += expanded_map[und_parts[0]] + expanded_map[und_parts[1]]
    
    return total + current


