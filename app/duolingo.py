
import re

def roman_to_int(s):
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
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000
    }
    
    words = s.lower().replace('-', ' ').split()
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
                total += current * value
                current = 0
            else:
                current += value
    
    return total + current

def detect_language_and_convert(s):
    # Check if it's a Roman numeral
    if re.match(r'^[IVXLCDM]+$', s.upper()):
        return roman_to_int(s.upper()), 0  # priority 0 for Roman
    
    # Check if it's an Arabic numeral
    if re.match(r'^\d+$', s):
        return int(s), 5  # priority 5 for Arabic
    
    # Check if it's English
    english_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
                    'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
                    'hundred', 'thousand', 'million']
    
    if any(word in s.lower() for word in english_words):
        return english_to_int(s), 1  # priority 1 for English
    
    # Check if it's German
    german_words = ['null', 'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben',
                   'acht', 'neun', 'zehn', 'elf', 'zwölf', 'dreizehn', 'vierzehn',
                   'fünfzehn', 'sechzehn', 'siebzehn', 'achtzehn', 'neunzehn', 'zwanzig',
                   'dreißig', 'vierzig', 'fünfzig', 'sechzig', 'siebzig', 'achtzig',
                   'neunzig', 'hundert', 'tausend', 'und']
    
    if any(word in s.lower() for word in german_words):
        return german_to_int(s), 4  # priority 4 for German
    
    # Check if it's Chinese
    chinese_chars = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                    '百', '千', '万', '億', '亿', '萬']
    
    if any(char in s for char in chinese_chars):
        # Distinguish between traditional and simplified
        if any(char in s for char in ['萬', '億']):
            return chinese_to_int(s), 2  # priority 2 for Traditional Chinese
        else:
            return chinese_to_int(s), 3  # priority 3 for Simplified Chinese
    
    # Default case
    try:
        return int(s), 5
    except:
        return 0, 5

def create_german_number_map():
    """Create a comprehensive map of German number words from 1 to 3999"""
    german_map = {}
    
    # Basic numbers
    basics = {
        0: 'null', 1: 'eins', 2: 'zwei', 3: 'drei', 4: 'vier', 5: 'fünf',
        6: 'sechs', 7: 'sieben', 8: 'acht', 9: 'neun', 10: 'zehn',
        11: 'elf', 12: 'zwölf', 13: 'dreizehn', 14: 'vierzehn', 15: 'fünfzehn',
        16: 'sechzehn', 17: 'siebzehn', 18: 'achtzehn', 19: 'neunzehn',
        20: 'zwanzig', 30: 'dreißig', 40: 'vierzig', 50: 'fünfzig',
        60: 'sechzig', 70: 'siebzig', 80: 'achtzig', 90: 'neunzig'
    }
    
    # Add basic numbers
    for num, word in basics.items():
        if num <= 3999:
            german_map[word] = num
    
    # Generate numbers 21-99
    for tens in [20, 30, 40, 50, 60, 70, 80, 90]:
        for ones in range(1, 10):
            if tens + ones <= 99:
                word = f"{basics[ones]}und{basics[tens]}"
                german_map[word] = tens + ones
    
    # Generate hundreds (100-999)
    for hundreds in range(1, 10):
        hundred_word = f"{basics[hundreds]}hundert"
        german_map[hundred_word] = hundreds * 100
        
        # Add numbers like 101, 102, etc.
        for ones in range(1, 10):
            word = f"{basics[hundreds]}hundert{basics[ones]}"
            german_map[word] = hundreds * 100 + ones
        
        # Add numbers like 110, 111, etc.
        for tens_ones in range(10, 100):
            if tens_ones in basics:
                word = f"{basics[hundreds]}hundert{basics[tens_ones]}"
                german_map[word] = hundreds * 100 + tens_ones
            elif 20 <= tens_ones < 100 and tens_ones % 10 == 0:
                word = f"{basics[hundreds]}hundert{basics[tens_ones]}"
                german_map[word] = hundreds * 100 + tens_ones
            elif 21 <= tens_ones <= 99:
                ones_digit = tens_ones % 10
                tens_digit = tens_ones - ones_digit
                word = f"{basics[hundreds]}hundert{basics[ones_digit]}und{basics[tens_digit]}"
                german_map[word] = hundreds * 100 + tens_ones
    
    # Generate thousands (1000-3999)
    for thousands in range(1, 4):  # Only up to 3 for 3000-3999
        thousand_word = f"{basics[thousands]}tausend"
        german_map[thousand_word] = thousands * 1000
        
        # Add numbers like 1001, 1002, etc.
        for ones in range(1, 10):
            word = f"{basics[thousands]}tausend{basics[ones]}"
            german_map[word] = thousands * 1000 + ones
        
        # Add numbers like 1010, 1011, etc.
        for tens_ones in range(10, 100):
            if tens_ones in basics:
                word = f"{basics[thousands]}tausend{basics[tens_ones]}"
                german_map[word] = thousands * 1000 + tens_ones
        
        # Add numbers with hundreds
        for hundreds in range(1, 10):
            # Numbers like 1100, 1200, etc.
            word = f"{basics[thousands]}tausend{basics[hundreds]}hundert"
            german_map[word] = thousands * 1000 + hundreds * 100
            
            # Numbers like 1101, 1102, etc.
            for ones in range(1, 10):
                word = f"{basics[thousands]}tausend{basics[hundreds]}hundert{basics[ones]}"
                german_map[word] = thousands * 1000 + hundreds * 100 + ones
    
    # Add some common variants and alternative spellings
    variants = {
        'ein': 1, 'eine': 1, 'einen': 1, 'einem': 1, 'einer': 1,
        'zwo': 2,  # Alternative for zwei
        'siebzehn': 17, 'siebzig': 70  # Alternative spellings
    }
    
    for variant, num in variants.items():
        if num <= 3999:
            german_map[variant] = num
    
    return german_map

# Create the German number map once
GERMAN_NUMBER_MAP = create_german_number_map()

def german_to_int(s):
    """Look up German number words in the precomputed map"""
    # Normalize the input
    normalized = s.lower().replace('und', 'und').replace('-', '').replace(' ', '')
    
    # Try exact match first
    if normalized in GERMAN_NUMBER_MAP:
        return GERMAN_NUMBER_MAP[normalized]
    
    # Try to find the closest match (handle some variations)
    for german_word, value in GERMAN_NUMBER_MAP.items():
        if german_word in normalized or normalized in german_word:
            return value
    
    # If no match found, try to parse it (fallback)
    return parse_german_fallback(normalized)

def parse_german_fallback(s):
    """Fallback parser for German numbers (simplified version)"""
    word_to_num = {
        'eins': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 'fünf': 5,
        'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
        'elf': 11, 'zwölf': 12, 'hundert': 100, 'tausend': 1000
    }
    
    # Simple regex-based parsing as fallback
    total = 0
    current = 0
    
    for word in word_to_num:
        if word in s:
            value = word_to_num[word]
            if value == 100:
                if current == 0:
                    current = 1
                current *= value
            elif value == 1000:
                if current == 0:
                    current = 1
                total += current * value
                current = 0
            else:
                current += value
    
    return total + current

def chinese_to_int(s):
    char_to_num = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000, '万': 10000, '億': 100000000, '亿': 100000000
    }
    
    # Handle traditional Chinese characters
    s = s.replace('萬', '万').replace('億', '亿')
    
    total = 0
    current = 0
    prev_multiplier = 1
    
    for char in s:
        if char in char_to_num:
            value = char_to_num[char]
            
            if value < 10:  # Digit (0-9)
                current = value
            elif value < 10000:  # Multiplier (十, 百, 千)
                if current == 0:
                    current = 1
                total += current * value
                current = 0
                prev_multiplier = value
            else:  # Large multiplier (万, 亿)
                if current == 0:
                    current = 1
                total = (total + current) * value
                current = 0
    
    return total + current

# Test the specific cases
def test_parsers():
    test_cases = [
        "四十五",  # Should be 45
        "XLIX",    # Should be 49
        "siebenundachtzig",  # Should be 87
        "one hundred twenty",  # Should be 120
        "dreihundertelf",  # Should be 311
        "one thousand one hundred",  # Should be 1100
        "MCMXCIX",  # Should be 1999
        "五萬四千三百二十一",  # Should be 54321
        "100000"    # Should be 100000
    ]
    
    for case in test_cases:
        value, priority = detect_language_and_convert(case)
        print(f"{case} -> {value} (priority {priority})")

def main():
    print(test_parsers())

if __name__ == "__main__":
    main()