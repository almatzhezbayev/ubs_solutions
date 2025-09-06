
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

# Add this test case function to your code
def test_large_dataset():
    """Test with a large dataset of mixed language numbers"""
    large_test_cases = [
        "167763935", "three hundred sixty-two million one hundred thirty-two thousand five hundred fifty-four",
        "九千零一十万七千八百三十一", "九亿四千二百三十二万五千九百九十五",
        "六億二千一百九十四萬零九十三", "dreihundertsechsundvierzig millionen sechshundertfünfundachtzigtausenddreihundertzwanzig",
        "三亿三千四百七十五万三千五百八十六", "二億五千二百一十五萬零九百一十",
        "six hundred seventy-eight million four hundred seven thousand five hundred eighty-seven", "zweihundertachtzig millionen dreihunderteinundzwanzigtausendfünfhundertsiebenundvierzig",
        "838264614", "八亿一千七百九十六万四千零一十",
        "688532539", "四億八千二百一十五萬零九百九十三",
        "one hundred forty-nine million eight hundred thirty-four thousand nine hundred sixty-nine", "七亿六千三百六十四万零三百二十二",
        "achthunderteinundfünfzig millionen neunhundertvierundachtzigtausendvierhunderteinundfünfzig", "一亿七千零八万九千七百二十八",
        "three hundred thirty-one million three hundred seventy-three thousand six hundred eighty-five", "siebenhundertvier millionen vierhundertvierundachtzigtausenddreihundertsiebenundsechzig",
        "五亿三千一百五十六万零一百一十五", "zweihundertachtunddreißig millionen sechshundertfünfundvierzigtausendvierhundertsechsundsechzig",
        "七億五千七百八十七萬八千五百七十二", "九千零七十二万四千一百八十",
        "eight hundred sixteen million seven hundred two thousand eight hundred thirty-six", "三亿六千四百四十九万二千三百六十八",
        "986053357", "二亿四千六百六十四万九千七百三十六",
        "六亿六千一百四十一万二千三百五十七", "fünfundfünfzig millionen vierhundertsiebzehntausendeinhunderteinundzwanzig",
        "achthundertdreiundvierzig millionen achthundertsiebenundvierzigtausendachthundertfünfundvierzig", "482140941",
        "259210448", "八億六千九百六十七萬一千一百六十九",
        "五亿九千五百四十六万一千三百零七", "三亿九千二百四十一万九千四百七十七",
        "achthundertneun millionen dreihundertzweiunddreißigtausendsechshundertfünfzig", "七億八千七百五十五萬三千七百一十四",
        "二亿零六百四十万一千五百三十九", "siebenhunderteinunddreißig millionen sechshundertsiebenundsiebzigtausendsechshundertneunundsiebzig",
        "一億六千零七十九萬五千三百七十八", "八亿六千四百五十九万四千五百五十八",
        "five hundred seven million one hundred thirty-three thousand seven hundred forty-eight", "八亿五千八百八十八万一千一百零七",
        "siebenhundertneunzig millionen fünfhundertneunundneunzigtausendzweihundertzweiunddreißig", "七億七千零七十五萬九千四百四十一",
        "一億六千四百二十五萬八千三百一十一", "一千三百二十七万零三百五十六",
        "二亿四千二百三十九万六千三百三十五", "五億一千五百五十二萬八千一百八十九",
        "三亿一千二百九十六万八千六百一十四", "七千八百三十六万六千三百四十八",
        "七亿二千五百八十六万九千七百九十五", "四億零四百八十五萬四千八百六十二",
        "one hundred eighty-eight million seven hundred sixty-one thousand three hundred forty-five", "二亿二千九百六十五万三千零六十九",
        "二億零六百二十九萬一千九百五十七", "946498747",
        "九億零九百六十二萬二千四百八十八", "七亿一千六百九十三万零五百五十九",
        "六亿四千一百二十七万一千七百八十八", "三亿五千六百零七万五千六百四十七",
        "248822854", "八億三千八百九十五萬八千三百六十九",
        "217923121", "397373826",
        "四億四千四百四十二萬一千六百一十五", "四千一百五十七万五千二百七十一",
        "六億四千六百九十一萬零四百七十九", "三億二千五百八十六萬九千六百四十八",
        "132544161", "二億四千七百六十七萬九千零三十六",
        "三千一百一十四万三千六百八十五", "八億三千二百八十二萬七千九百二十一",
        "五億五千九百四十八萬零八百八十一", "siebenhunderteinunddreißig millionen einhundertsiebenundneunzigtausendeinundvierzig",
        "八億五千零六十九萬零七百四十四", "sechshundertsechzig millionen sechshunderteinundneunzigtausendsechshundertfünfundachtzig",
        "2949504", "二億四千四百九十二萬四千三百八十九",
        "neunhundertdreißig millionen vierhunderteinundachtzigtausendneunhundertvierzehn", "九千三百四十四万二千五百四十三",
        "九亿九千六百四十九万五千六百二十", "四億六千二百三十萬七千零六十八",
        "zweihundertacht millionen neunhundertachtundachtzigtausendsiebenhundertsechsunddreißig", "748567899",
        "422482200", "699114855",
        "217230214", "二亿六千六百九十四万九千三百六十二",
        "八億三千三百九十七萬六千四百五十八", "16503377",
        "199747002", "ninety-nine million nine hundred fifty-one thousand three hundred forty-three",
        "four hundred ninety-five million one hundred seventy-two thousand five hundred forty", "seven hundred eighty-three million two hundred thirty-seven thousand seven hundred thirty",
        "301874256", "195458916",
        "four hundred ninety-seven million three hundred forty-one thousand seven hundred twenty-one", "five hundred eighty-four million nine hundred ninety-four thousand nine hundred eight",
        "dreihundertsieben millionen neunzehntausendachthundertachtundzwanzig", "九亿七千四百四十六万八千七百一十九",
        "八亿四千五百五十一万六千七百六十五", "516703709",
        "neunhunderteinundachtzig millionen siebenhundertfünfundachtzigtausendsiebenhundertsieben", "sechshundertvierunddreißig millionen fünfhundertsechsundsechzigtausendfünfhundertvierundsechzig",
        "六亿零二十三万三千六百一十五", "siebenhundertsiebenundvierzig millionen achthundertfünfundfünfzigtausendfünfhundertvierzig",
        "784429348", "fünfhundertvierzehn millionen siebenundfünfzigtausendzweihundertvierunddreißig",
        "two hundred eleven million two hundred twenty-one thousand six hundred fourteen", "七千一百七十九萬六千九百三十七",
        "九亿六千六百四十二万七千九百三十五", "zweihundertneun millionen zweihundertdreiundneunzigtausendneunhundertzwei",
        "519972428", "六亿三千九百二十七万四千九百六十四",
        "883219760", "fünfhundertzwei millionen sechshundertvierzehntausendeinhunderteinundvierzig",
        "五亿五千一百七十万九千一百二十一", "六亿二千九百九十四万三千八百四十五",
        "八億零四百六十三萬三千一百四十八", "nine hundred sixty-nine million one hundred sixty-seven thousand two hundred ninety-nine",
        "七億一千八百七十四萬二千四百一十七", "399619190",
        "two hundred twenty-four million ninety-six thousand one hundred ninety-six", "430143222",
        "四亿四千三百四十三万六千五百五十九", "二亿五千六百七十一万四千一百七十九",
        "four hundred nine million one hundred eighty-six thousand seven hundred sixty-eight", "三億三千五百一十三萬四千八百八十一",
        "794999110", "sechshundertdreiundsiebzig millionen siebenhundertachtundneunzigtausendeinhundertsechsundsechzig",
        "853002966", "866901651",
        "七亿六千七百五十一万六千八百三十九", "三億二千零三十六萬八千二百七十一",
        "九億一千四百九十七萬三千五百一十八", "four hundred eight million seven hundred seventy-four thousand nine hundred fifty-three",
        "two hundred fifty-two million nine hundred one thousand one hundred forty", "五亿零一百四十九万五千八百零二",
        "208494659", "524267143",
        "neunhundertzweiundsechzig millionen sechshundertneunundzwanzigtausendeinhundertachtundneunzig", "五億七千三百七十三萬三千二百八十二",
        "二亿二千五百万八千一百八十七", "zweihundertzweiundsiebzig millionen siebenhundertsechzehntausendeinhundertsechsundvierzig",
        "七億零二百七十萬四千八百零八", "two hundred forty-six million three hundred thirty-two thousand nine hundred twenty-seven",
        "一億零三十三萬二千九百六十七", "two hundred thirty-six million three hundred four thousand two hundred forty-four",
        "two hundred fifty-six million two hundred fifty-eight thousand eighteen", "fünfhundertfünfundsiebzig millionen zweihundertvierundzwanzigtausendneunhundertneunundsiebzig",
        "293721316", "二亿三千六百四十六万三千九百九十",
        "vierhundertelf millionen vierhundertdreiundsiebzigtausendsechshundertneununddreißig", "八亿三千六百二十七万八千六百二十七",
        "一億八千零九十萬六千九百八十九", "七億七千二百六十九萬五千三百四十八",
        "84291103", "neunhundertfünfundsechzig millionen dreihundertdreiundzwanzigtausendzweihundertneunundsiebzig",
        "一亿五千五百六十八万八千零九", "一亿九千四百九十万二千一百五十八",
        "nine hundred forty-five million fifty-one thousand six hundred ninety-two", "七億五千六百六十萬二千五百二十九",
        "一億九千八百一十萬八千五百五十一", "一亿五千三百六十五万四千四百九十六",
        "zwei millionen achthundertsiebentausendfünfhundertacht", "三亿三千四百一十万三千二百六十",
        "一亿一千二百一十一万五千一百一十二", "42571418",
        "vierhundertzwölf millionen zweihundertfünfundsiebzigtausenddreihundertvierundsiebzig", "八亿一千六百一十八万零九百二十九",
        "forty-four million four hundred eleven thousand nine hundred nineteen", "fünfhundertzweiundfünfzig millionen achthundertvierundzwanzigtausendsiebenhundertneun",
        "二億零八百零七萬一千七百一十四", "四億九千六百七十萬四千六百三十四",
        "一億七千五百九十三萬一千五百五十六", "948636588",
        "einhundertsiebenundsiebzigtausendfünfhundertzweiundzwanzig", "八亿八千零五十一万五千七百七十三",
        "sixty-four million nine hundred fifty-eight thousand five hundred fifty-five", "104871545",
        "586055038", "eight hundred eighty-eight million four hundred thirty-seven thousand eight hundred twenty-five",
        "eight hundred eighty-nine million eight hundred eighty-two thousand two hundred eighty-one", "fünfhundertsechsunddreißig millionen siebenhundertvierundsiebzigtausendneunhundertachtunddreißig",
        "two hundred eighty-eight million three hundred ninety-five thousand two hundred sixteen", "einhundertneunundneunzig millionen zweihundertsiebzigtausendzweihundertdrei",
        "two hundred thirty-three million thirteen thousand two hundred eighty-three", "一億九千九百六十九萬九千六百四十八",
        "vierhundertzwölf millionen zweihundertsiebzigtausendzweihundertzweiundfünfzig", "neunhundertsiebenundsechzig millionen neunhundertsechsundsiebzigtausendsiebenhunderteins",
        "535094256", "seven hundred thirty-three million nine hundred sixty-four thousand one hundred twenty-nine",
        "793100575", "six hundred twenty-one million eight hundred twelve thousand three hundred eight",
        "nine hundred twenty-six million two hundred thirty-eight thousand three hundred twenty-eight", "五千一百六十六万三千八百四十九",
        "二亿一千零三万零四百四十八", "dreihundertfünfundsechzig millionen einhundertachtundvierzigtausendsiebenundsiebzig",
        "一億九千零六萬一千三百八十一", "八亿六千八百四十二万八千一百一十二",
        "一億零四百七十二萬九千四百零六", "achthundertsechsundfünfzig millionen einhunderteinundachtzigtausendfünfhundertsiebenundfünfzig",
        "一亿二千八百五十九万八千四百三十六", "五億二千三百零九萬零一百六十七",
        "一亿六千八百八十四万零八百五十", "717826367",
        "3323031", "四亿零六百三十九万零四百二十",
        "一億五千零四十四萬一千八百九十五", "einhundertfünfzehn millionen einhundertzweiundfünfzigtausendachthundertfünfundzwanzig",
        "siebenhundertsiebenundzwanzig millionen siebenhundertsechsundvierzigtausendzweihundertdreißig", "548377022",
        "84877028", "186959952",
        "five hundred fifty million four hundred six thousand six hundred ninety-five", "zweihundertvierzig millionen dreihundertsiebenundsechzigtausenddreihundertfünfzehn",
        "fünfundachtzig millionen sechsundfünfzigtausendachthundertfünfundzwanzig", "499771749",
        "neunhundertvierundsiebzig millionen zweihundertsiebenundzwanzigtausendfünfhundertsechsundsechzig", "四億九千一百六十五萬四千一百五十六",
        "544701061", "六億零九十六萬六千五百四十九",
        "五百八十九萬零七百一十六", "289819587",
        "neunhundertzehn millionen neunhundertachttausendachthundertsechzig", "one hundred two million six hundred eighty-seven thousand two hundred seventy-three",
        "一億九千五百七十四萬九千二百一十六", "430859723",
        "seven hundred eighty-six million seven hundred ninety-seven thousand seven hundred fifty-two", "227697686",
        "einhundertzehn millionen fünfhundertachtzigtausendvierhundertvierundneunzig", "25955731",
        "九億四千五百一十五萬二千零一十二", "三亿八千四百四十万八千五百七十",
        "siebenhundertvierundzwanzig millionen siebenhundertdreiundfünfzigtausendsechshundertsiebenundsechzig", "vierhundertsechsundsiebzig millionen einhundertsechsundachtzigtausenddreihundertvierzehn",
        "四億五千三百五十三萬五千八百八十六", "one hundred seventeen million four hundred sixty-six thousand five hundred twenty",
        "一亿四千九百二十七万五千四百零四", "three hundred one million three hundred eighteen thousand sixty-three",
        "二亿零八百七十八万九千六百八十三", "fünfhundertsechzig millionen siebenhundertsiebenundsechzigtausendachthunderteinundzwanzig",
        "dreihundertsechsundvierzig millionen sechshundertsechsundzwanzigtausendsiebenundsechzig", "二億六千零三十萬九千二百四十二",
        "七亿一千七百万三千六百三十七", "四亿五千四百九十七万九千九百六十七",
        "九亿三千五百二十八万二千五百五十七", "五亿九千七百一十九万八千八百",
        "sechshundertsiebenundneunzig millionen siebenhundertfünfzigtausenddreihundertsechsunddreißig", "七億一千六百九十一萬零八百二十一",
        "zweihundertsiebzehn millionen dreihundertsiebenundsechzigtausendfünfhundertvierzehn", "四亿一千一百零四万四千四百三十六",
        "704958436", "neunhundertdreiundzwanzig millionen vierhundertdreiundachtzigtausendzweihundertsechsundsechzig",
        "八千五百一十六万五千七百三十二", "九億六千八百八十八萬五千三百七十",
        "八亿一千二百零一万五千九百一十六", "七億三千九百三十萬五千七百六十四",
        "two hundred sixty-nine million two hundred four thousand seven hundred fifty-one", "fifty-nine million eight hundred fourteen thousand seven hundred sixty-one",
        "九亿七千七百六十四万三千零八十四", "二億七千六百一十五萬七千九百六十九",
        "七亿零七百九十万五千一百八十三", "一億六千二百二十萬二千二百三十七",
        "八億九千八百九十四萬九千五百九十九", "一億四千八百一十一萬零七百四十三",
        "siebenhundertzweiunddreißig millionen sechshunderteinundvierzigtausendsechshundert", "296835653",
        "535445113", "四亿二千九百九十三万六千四百五十三",
        "195049914", "einhundertvierunddreißig millionen vierhunderteinstausendsiebenhundertzweiunddreißig",
        "二億零三百二十萬九千二百八十二", "七亿四千一百八十八万二千三百四十三",
        "six hundred sixty-nine million nine hundred nineteen thousand nine hundred one", "neunhundertvierzig millionen zweihundertvierundsechzigtausenddreihunderteinundzwanzig",
        "四億二千一百一十六萬五千七百", "zweihundertdreiundachtzig millionen zweihundertachtundvierzigtausendzweihundertdreiundneunzig",
        "六亿七千九百八十万三千零一十一", "六亿七千二百九十二万八千零八十六",
        "two hundred eight million three hundred eighty-three thousand five hundred twelve", "三億二千八百六十九萬七千零二十六",
        "six hundred twenty-three thousand eight hundred forty", "sechzehn millionen fünfhundertneunundneunzigtausendsechshunderteinundzwanzig",
        "九億七千零四十二萬五千二百四十四", "二億零四百五十四萬一千二百零九",
        "siebenhundertdreiundneunzig millionen siebenhundertsechsundneunzigtausendsechshundertsiebenundzwanzig", "二千七百七十九万六千五百六十九",
        "zweihundertzwölf millionen fünfhundertelftausendelf", "六億二千五百二十六萬五千四百五十九",
        "siebenhundertsechsundsechzig millionen sechshundertdreiundneunzigtausendsechshundertsechsundzwanzig", "zweihunderteinundsiebzig millionen fünfhundertachtundfünfzigtausendsechshundertfünfzehn",
        "five hundred thirty-one million eight hundred eighty-eight thousand six hundred sixty-one", "sechshundertvierzig millionen fünfundachtzigtausendneunhundertfünfundneunzig",
        "three hundred twenty-three million seven hundred thousand two hundred forty-three", "七亿零一百零九万八千四百一十一",
        "73536031", "五億一千四百八十五萬五千七百九十二",
        "eight hundred fifty-nine million six hundred eighty thousand four hundred thirty-five", "925187832",
        "八亿六千八百七十二万九千九百六十", "四千九百七十八萬五千四百一十四",
        "九億八千三百三十八萬七千九百九十三", "one hundred twenty-nine million three hundred seventy-six thousand five hundred eighty-six",
        "eight hundred eighty-nine million four hundred ninety-one thousand six hundred fifty-six", "sechshundertzweiunddreißig millionen einhundertzwölftausendfünfhundertneununddreißig",
        "247248433", "三千八百四十四萬九千八百零九",
        "einhundertneunzehn millionen zweihundertfünfundfünfzigtausenddreihundertneun", "neunhundertfünfzehn millionen achthundertvierunddreißigtausendfünfhunderteinundsechzig",
        "二億四千三百一十萬七千四百零八", "377714052",
        "68825386", "一亿五千零三十一万零一百一十五",
        "三亿四千零三十四万九千四百七十", "七亿零五百四十二万九千六百四十三",
        "643058243", "vierhundertzwei millionen sechzigtausendsechshundertdrei",
        "siebenhundertsechzig millionen vierhundertfünfundzwanzigtausendneunhundertvier", "fünfhundertzwanzig millionen siebenhundertzweiundsiebzigtausendeinhundertfünfundzwanzig",
        "nine hundred ten million four hundred ninety-one thousand six hundred ninety-two", "五亿一千三百九十八万五千一百二十七",
        "one hundred sixty-two million seven hundred one thousand two hundred fifty-seven", "one hundred six million ninety-eight thousand seventy-three",
        "neunhundertachtunddreißig millionen vierhundertneunzehntausendneunhundertsechsundsechzig", "four hundred sixteen million seven hundred sixty-five thousand three hundred sixty-four",
        "八亿七千六百五十三万零二百三十九", "795279545",
        "七億四千零四十七萬六千一百九十一", "五亿四千八百五十八万三千八百三十三",
        "neunhundertneunundvierzig millionen dreihundertsechsundzwanzigtausendsiebenhundertneunzig", "neunhundertneunundfünfzig millionen zweihundertachtundvierzigtausendzweihundertneunundachtzig",
        "二亿四千八百八十八万二千八百零五", "四亿六千一百三十二万五千三百九十一",
        "267381946", "196872585",
        "sechshundertdreiundsechzig millionen fünfhundertsechsundneunzigtausenddreihundertdreiundvierzig", "二億四千一百三十八萬三千一百四十一",
        "六億零三十七萬四千九百八十五", "六亿六千四百六十六万零五百二十一",
        "三億三千三百六十七萬六千五百二十一", "一億一千一百九十四萬六千九百五十",
        "三亿二千四百二十五万零七百一十", "五千五百二十六萬三千九百七十一",
        "two hundred thirty-two million five hundred eighty-three thousand six hundred ninety-three", "五亿四千二百七十二万三千九百五十一",
        "657664078", "二亿零九十八万八千二百四十三",
        "siebenhundertdreiunddreißig millionen einhundertsiebzehntausendzweihundertvierundachtzig", "one hundred ninety-two million five hundred seventy-six thousand four hundred ninety-six",
        "三亿零五百七十四万一千九百四十", "eight hundred thirty-eight million seven hundred ninety-one thousand four hundred three",
        "279039315", "five hundred seven million nine hundred seventy thousand one hundred eighty-three",
        "七億八千四百七十五萬一千一百四十六", "629494480",
        "zweihundertfünf millionen vierhundertdreiundzwanzigtausendsiebenhundertdreiundachtzig", "八億三千六百二十一萬四千一百二十七",
        "621365917", "one hundred thirty-one million five hundred eleven thousand five hundred thirty-one",
        "一亿一千六百零四万零二百六十八", "654196967",
        "152968993", "872133434",
        "einhundertneunundachtzig millionen fünfundsiebzigtausendfünfhundertneunundzwanzig", "七億零八百五十八萬二千七百五十五",
        "two hundred million six hundred ninety thousand five hundred twelve", "五亿九千八百四十六万九千五百三十九",
        "dreihundertfünfzehn millionen eintausendvierhundertneunundneunzig", "五億七千八百五十三萬八千五百九十八",
        "five hundred thirty-four million six hundred ten thousand twenty-four", "nine hundred eighty-nine million two hundred fifty-nine thousand four hundred twenty-eight",
        "four hundred seventy-two million seven hundred twenty-three thousand two hundred fifty-seven", "八千六百零四万六千七百零八",
        "207712385", "六億八千四百六十七萬二千一百二十五",
        "zweihundertdreiundsiebzig millionen vierhundertsiebenundfünfzigtausendsechshundertdrei", "dreihundertachtunddreißig millionen zweihundertneununddreißigtausendsechshundertzweiundachtzig",
        "九亿二千零一十四万二千零三十二", "二千八百七十五萬四千五百三十六",
        "四億零七百七十七萬八千零三十七", "two hundred fifty-one million one hundred fifty-three thousand eighty-eight",
        "five hundred two million seven hundred seventy-five thousand one hundred sixteen", "fünfhundertvierundachtzig millionen sechshundertsiebzehntausendzweihundertsechsundachtzig",
        "vierhundertachtundvierzig millionen einhundertdreiundsiebzigtausendsechshundertzweiunddreißig", "四億三千二百八十八萬九千四百三十九",
        "六百零五万一千五百七十五", "三億零六百三十六萬八千零六十七",
        "五亿九千九百二十五万零八百八十二", "four hundred fifty-nine million four hundred sixty-five thousand six hundred ninety-two",
        "303545915", "225024354",
        "seven hundred eighteen million eight hundred fourteen thousand four hundred fifty-three", "dreihundertfünfunddreißig millionen fünfhundertdreitausendsechshundertachtundfünfzig",
        "594130858", "七亿四千九百七十七万九千六百一十一",
        "789272900", "sechshundertsiebenundsechzig millionen vierhundertachtundvierzigtausendfünfhundertsiebzig",
        "772474842", "794159745",
        "六亿三千二百三十三万零八百八十五", "481392266",
        "六千七百九十四万三千二百七十七", "390273089",
        "五億八千零二十七萬二千七百九十六", "八亿一千七百四十五万九千七百六十一",
        "一億六千三百零七萬零八百四十二", "九千六百四十九万八千五百八十",
        "siebenundachtzig millionen zweihundertachtundzwanzigtausendzweihundertsechzig", "290682481",
        "六亿四千四百一十四万六千三百四十二", "elf millionen zweihunderteinundneunzigtausendsechshundertvierundsechzig",
        "300040314", "84634128",
        "vierhunderteinundfünfzig millionen dreihundertdreißigtausendzweiundsechzig", "eight hundred eleven million five hundred twenty thousand nine hundred seventy-four",
        "fünfhundertvierundzwanzig millionen fünfhundertvierunddreißigtausendachthundertvierundzwanzig", "二億五千四百三十九萬一千八百二十三",
        "nine hundred sixty-seven million five hundred thirty-nine thousand thirty-nine", "五亿八千五百二十九万一千一百四十七",
        "一亿九千四百五十五万九千四百一十八", "五亿七千三百一十六万零五百九十五",
        "three hundred two million two hundred seventy-four thousand four hundred seventy-nine", "二億五千三百七十九萬六千零七十六",
        "一億七千七百五十六萬八千二百三十六", "fünfhundertdreißig millionen vierhundertdreiundzwanzigtausendfünfhundertzwölf",
        "856163220", "553336717",
        "sechshundertdreiundzwanzig millionen vierhundertsiebenundachtzigtausendsechshundertfünfundachtzig", "九百零四万零一百八十二",
        "532566463", "879199843",
        "一億四千四百六十二萬五千零四十八", "six hundred ten million five hundred three thousand three hundred nineteen",
        "七亿二千四百零三万六千七百零四", "siebenhundertsechsundzwanzig millionen vierhundertfünfundfünfzigtausendfünfhunderteinunddreißig",
        "六亿三千八百五十九万二千二百六十", "one hundred forty-seven million fifty-five thousand eight hundred one",
        "七亿四千九百五十六万四千一百一十六", "four hundred thirty-four million five hundred ninety-three thousand seven hundred thirty-one",
        "two hundred sixty-five million one hundred five thousand nine hundred ninety", "七億五千二百四十三萬九千零一十",
        "三亿七千七百二十二万六千九百二十八", "七亿九千四百八十三万一千八百七十三",
        "einhundertsiebenundvierzig millionen dreihundertdreiundsiebzigtausenddreihundertachtundachtzig", "six hundred eighty-four million three hundred forty thousand two hundred eighteen",
        "二億八千八百四十六萬五千八百三十九", "二亿七千二百一十六万七千三百八十五",
        "七亿八千三百六十五万二千九百四十五", "five hundred thirteen million four hundred four thousand seventy-seven",
        "seven hundred ninety-one million six hundred twenty-six thousand four hundred ninety-four", "七千九百一十九萬二千八百零八",
        "three million eight hundred twenty thousand five hundred forty-three", "691710780",
        "972722754", "988601531",
        "六億三千八百九十萬一千零一十八", "568693838",
        "九亿七千八百零三万三千九百一十三", "157321517",
        "四亿六千六百零六万零八百六十一", "vierhundertneunundsiebzig millionen fünfhundertvierzehntausendvierhundertzweiundsiebzig",
        "427697137", "495364939",
        "549473979", "two hundred forty-six million eight hundred twenty-four thousand seven hundred sixty-five",
        "二億一千二百五十七萬七千八百二十二", "one hundred sixty-five million one hundred seven thousand six hundred five",
        "三億五千四百七十八萬三千四百七十二", "二亿一千九百二十三万八千六百九十六",
        "five hundred forty-nine million one hundred forty-two thousand eight hundred sixty", "700440036",
        "sechsundzwanzig millionen neunhundertachtundsechzigtausendzweihundertvierundachtzig", "二亿一千七百二十六万五千四百二十四",
        "230063984", "220574103",
        "achtundfünfzig millionen vierhundertfünfunddreißigtausendsechshundertfünf", "einunddreißig millionen siebenhundertsiebentausendfünfhundertzweiunddreißig",
        "neunhundertzwölf millionen sechshundertvierundvierzigtausendachthundertachtundsechzig", "三億二千四百九十二萬一千六百八十八",
        "siebenhundertneunzehn millionen achthundertsechsundzwanzigtausendzweihundertvierundachtzig", "793779126",
        "786414059", "六千三百五十三万三千九百三十九",
        "504951468", "nine hundred twenty-four million nine hundred sixty-nine thousand seven hundred eighty-four",
        "five hundred sixty-four million seven hundred eighty-nine thousand nine hundred sixty-five", "229187716",
        "vierhundert millionen achthundertachtundfünfzigtausendsiebenundneunzig", "639308277",
        "92113382", "八億五千六百六十二萬四千二百八十八",
        "fünfhundertsiebzig millionen vierhundertzweiunddreißigtausendvierhundertzweiundvierzig", "71867843",
        "five hundred seventy-eight million five hundred fifty-eight thousand forty-two", "九億六千九百三十八萬五千三百一十一",
        "一亿七千零四十五万五千二百六十六", "一億八千五百二十三萬三千六百二十六",
        "476738666", "96740357",
        "155878051", "七億一千四百六十七萬零六百三十五",
        "dreihundertelf millionen zweihundertviertausenddreihundertfünfzig", "one hundred eighty-four million eight hundred fifty-nine thousand five hundred fifty-seven",
        "seven hundred fifty-seven million six hundred fifty-one thousand one hundred ninety-seven", "272676290",
        "八千四百六十六万八千七百五十一", "six hundred seventy million seven hundred five thousand eighty-two",
        "九亿五千三百八十二万五千一百", "eight hundred forty-six million two hundred thirty-four thousand four hundred thirty-four",
        "six hundred fifty-four million six hundred forty-nine thousand six hundred sixteen", "one hundred seventeen million five hundred forty-one thousand seven hundred fifty-nine",
        "六億八千五百九十二萬六千零一十六", "sixty-three million four hundred eighty-seven thousand four hundred ninety-five",
        "sechshundertneununddreißig millionen neunhundertvierundsechzigtausendachthundertacht", "八亿零七百五十一万二千八百二十七",
        "545975188", "三億零八百四十五萬七千七百零四",
        "siebenhundertdreißig millionen fünfhundertzwölftausendfünfhundertsechsundachtzig", "four hundred sixty-two million seven hundred one thousand six hundred three",
        "278387103", "二億三千一百七十五萬四千二百五十三",
        "siebenhundertzwölf millionen zweihundertvierundzwanzigtausendvierhundertdreißig", "150808814",
        "八億三千九百五十六萬三千四百一十一", "七億六千五百二十二萬六千六百一十",
        "七亿五千四百四十一万九千九百三十九", "五億二千九百五十四萬九千九百七十三",
        "101606365", "二千一百七十七万八千六百七十二",
        "一亿零二百三十二万零三百一十九", "186455405",
        "一億三千一百七十三萬三千一百四十六", "594198635",
        "594518411", "sixty-eight million six hundred sixty-nine thousand eight hundred sixty-two",
        "825993775", "八億四千一百五十萬二千九百九十七",
        "三千五百零六万四千四百五十六", "837305096",
        "298369186", "three hundred six million three hundred sixty-one thousand three hundred seventy-six",
        "one hundred eighty-nine million three hundred thirty-two thousand seven hundred fifty-one", "five hundred twenty-two million three hundred eighty-three thousand three hundred nine",
        "seven hundred twenty-two million three hundred sixty thousand five hundred thirty-one", "五亿四千一百九十六万七千零七十一",
        "einhundertsechsundachtzig millionen achthundertneunundvierzigtausendzweihundertsechzehn", "九百一十九万零五百三十二",
        "dreihundertfünfundsiebzig millionen zweihundertdreiundachtzigtausenddreihundertdreiundfünfzig", "五亿九千万六千七百六十一",
        "siebenhundertachtzehn millionen sechshundertdreiundvierzigtausendzweihundertachtundsiebzig", "八千九百六十三萬零九百七十七",
        "three hundred ninety-four million one hundred forty-three thousand three hundred sixty-nine", "五億一千九百三十五萬五千八百四十六",
        "二億六千六百一十六萬四千九百六十七", "四億三千一百零七萬四千六百零四",
        "zweihundertachtundneunzig millionen vierhundertfünfzigtausenddreihundertachtzehn", "五亿八千七百三十八万一千七百二十七",
        "三億三千五百零二萬六千零八十九", "二亿二千九百四十六万一千一百八十三",
        "八億五千三百三十五萬九千七百七十一", "一亿二千九百八十六万三千三百八十五",
        "five hundred thirty-nine million eight hundred seven thousand eight hundred seventy-six", "六亿一千五百九十万二千四百八十二",
        "435068520", "zwanzig millionen dreihundertzweiundfünfzigtausendfünfhundertzwanzig"]
    
    for case in large_test_cases:
        value, priority = detect_language_and_convert(case)
        print(f"{case} -> {value} (priority {priority})")

def main():
    print(test_large_dataset())

if __name__ == "__main__":
    main()