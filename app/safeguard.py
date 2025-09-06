import re
import math
from typing import List, Tuple

# Transformation functions for Challenge 1
def mirror_words(x: str) -> str:
    """Reverse each word in the sentence"""
    return ' '.join([word[::-1] for word in x.split()])

def encode_mirror_alphabet(x: str) -> str:
    """Replace each letter with its mirror in the alphabet"""
    result = []
    for char in x:
        if char.isalpha():
            if char.islower():
                result.append(chr(219 - ord(char)))  # a=97, z=122 -> 219-97=122, 219-122=97
            else:
                result.append(chr(155 - ord(char)))  # A=65, Z=90 -> 155-65=90, 155-90=65
        else:
            result.append(char)
    return ''.join(result)

def toggle_case(x: str) -> str:
    """Switch uppercase to lowercase and vice versa"""
    return x.swapcase()

def swap_pairs(x: str) -> str:
    """Swap characters in pairs within each word"""
    words = x.split()
    result = []
    for word in words:
        swapped = []
        for i in range(0, len(word) - 1, 2):
            swapped.extend([word[i+1], word[i]])
        if len(word) % 2 == 1:
            swapped.append(word[-1])
        result.append(''.join(swapped))
    return ' '.join(result)

def encode_index_parity(x: str) -> str:
    """Rearrange each word: even indices first, then odd indices"""
    words = x.split()
    result = []
    for word in words:
        evens = [word[i] for i in range(0, len(word), 2)]
        odds = [word[i] for i in range(1, len(word), 2)]
        result.append(''.join(evens + odds))
    return ' '.join(result)

def double_consonants(x: str) -> str:
    """Double every consonant"""
    vowels = 'aeiouAEIOU'
    result = []
    for char in x:
        result.append(char)
        if char.isalpha() and char not in vowels:
            result.append(char)
    return ''.join(result)

# Inverse functions for Challenge 1
def inverse_mirror_words(x: str) -> str:
    return mirror_words(x)  # Same as forward

def inverse_encode_mirror_alphabet(x: str) -> str:
    return encode_mirror_alphabet(x)  # Same as forward (involution)

def inverse_toggle_case(x: str) -> str:
    return toggle_case(x)  # Same as forward (involution)

def inverse_swap_pairs(x: str) -> str:
    return swap_pairs(x)  # Same as forward (involution)

def inverse_encode_index_parity(x: str) -> str:
    """Inverse of encode_index_parity"""
    words = x.split()
    result = []
    for word in words:
        mid = math.ceil(len(word) / 2)
        evens = word[:mid]
        odds = word[mid:]
        reconstructed = []
        for i in range(len(word)):
            if i % 2 == 0:
                reconstructed.append(evens[i//2] if i//2 < len(evens) else '')
            else:
                reconstructed.append(odds[i//2] if i//2 < len(odds) else '')
        result.append(''.join(reconstructed))
    return ' '.join(result)

def inverse_double_consonants(x: str) -> str:
    """Remove doubled consonants"""
    vowels = 'aeiouAEIOU'
    result = []
    i = 0
    while i < len(x):
        result.append(x[i])
        if (i + 1 < len(x) and x[i] == x[i+1] and 
            x[i].isalpha() and x[i] not in vowels):
            i += 1  # Skip the duplicate
        i += 1
    return ''.join(result)

def parse_nested_transformations(transformation_str: str) -> List[str]:
    """Parse nested transformation functions into a list of function names"""
    # Handle nested functions like "encode_mirror_alphabet(double_consonants(x))"
    # We need to parse from inside out
    
    # Remove spaces and normalize
    clean_str = transformation_str.strip()
    
    # If it's just a simple function call, return it
    if clean_str.count('(') == 1 and clean_str.endswith('(x)'):
        return [clean_str.replace('(x)', '')]
    
    # For nested calls, we need to parse from innermost to outermost
    functions = []
    current = clean_str
    
    # Keep extracting the innermost function until we're done
    while '(' in current and current != 'x':
        # Find the innermost function call (one that contains only 'x' or another variable)
        # Work backwards from the innermost parentheses
        
        # Find all function calls
        import re
        
        # Pattern to match function_name(content)
        pattern = r'(\w+)\([^()]*\)'
        matches = list(re.finditer(pattern, current))
        
        if not matches:
            break
            
        # Find the innermost match (the one with the simplest content)
        innermost_match = None
        for match in matches:
            content = current[match.start():match.end()]
            # Check if this contains only x or simple content
            inner_content = content[content.find('(')+1:content.rfind(')')]
            if inner_content.strip() in ['x', ''] or not re.search(r'\w+\(', inner_content):
                innermost_match = match
                break
        
        if not innermost_match:
            # If we can't find a simple innermost function, just take the first one
            innermost_match = matches[0]
        
        # Extract the function name
        func_name = innermost_match.group(1)
        functions.append(func_name)
        
        # Replace this function call with 'x' for the next iteration
        before = current[:innermost_match.start()]
        after = current[innermost_match.end():]
        current = before + 'x' + after
        
        # If we've reduced it to just 'x', we're done
        if current.strip() == 'x':
            break
    
    return functions

def reverse_transformations(transformed_word: str, transformations: List[str]) -> str:
    """Apply transformations in reverse order, handling nested functions"""
    current_word = transformed_word
    
    # Process each transformation string
    all_functions = []
    for transform_str in transformations:
        # Parse nested functions from this transformation
        nested_funcs = parse_nested_transformations(transform_str)
        all_functions.extend(nested_funcs)
    
    # Apply all functions in reverse order
    for func_name in reversed(all_functions):
        if func_name == 'mirror_words':
            current_word = inverse_mirror_words(current_word)
        elif func_name == 'encode_mirror_alphabet':
            current_word = inverse_encode_mirror_alphabet(current_word)
        elif func_name == 'toggle_case':
            current_word = inverse_toggle_case(current_word)
        elif func_name == 'swap_pairs':
            current_word = inverse_swap_pairs(current_word)
        elif func_name == 'encode_index_parity':
            current_word = inverse_encode_index_parity(current_word)
        elif func_name == 'double_consonants':
            current_word = inverse_double_consonants(current_word)
    
    return current_word

# Challenge 2: Coordinate pattern analysis
def analyze_coordinates(coordinates: List[List[str]]) -> str:
    """Extract hidden parameter from coordinate pattern"""
    # Convert string coordinates to float pairs
    coords = []
    for coord_pair in coordinates:
        lat = float(coord_pair[0])
        lng = float(coord_pair[1])
        coords.append((lat, lng))
    
    # According to hints:
    # - View from different perspective (spatial relationships)
    # - Remove anomalies/outliers that disrupt harmony
    # - Authentic coordinates resemble something simple yet significant
    # - Reveals a number critical to encryption scheme
    
    # Calculate centroid
    centroid_lat = sum(lat for lat, lng in coords) / len(coords)
    centroid_lng = sum(lng for lat, lng in coords) / len(coords)
    
    # Calculate distances from centroid to identify outliers
    distances = []
    for i, (lat, lng) in enumerate(coords):
        dist = math.sqrt((lat - centroid_lat)**2 + (lng - centroid_lng)**2)
        distances.append((dist, i, lat, lng))
    
    # Sort by distance
    distances.sort()
    
    # Try removing outliers (those furthest from centroid)
    # Keep the coordinates that are closest to each other
    median_dist = distances[len(distances)//2][0]
    
    # Filter out coordinates that are too far from the median distance
    threshold = median_dist * 1.5
    filtered_coords = []
    for dist, i, lat, lng in distances:
        if dist <= threshold:
            filtered_coords.append((lat, lng))
    
    # Check if remaining coordinates form a recognizable pattern
    if len(filtered_coords) >= 3:
        # Check if they form a geometric shape
        # For simplicity, let's see if they form a triangle or square
        
        # Method 1: Check for patterns in coordinate digits
        pattern_digits = []
        for lat, lng in filtered_coords:
            # Extract meaningful digits from coordinates
            lat_int = int(abs(lat))
            lng_int = int(abs(lng))
            
            # Look for single-digit patterns
            if lat_int < 10:
                pattern_digits.append(str(lat_int))
            if lng_int < 10:
                pattern_digits.append(str(lng_int))
        
        if pattern_digits:
            return ''.join(pattern_digits)
        
        # Method 2: Use the count of filtered coordinates
        return str(len(filtered_coords))
    
    # Method 3: Look for patterns in decimal places
    decimal_sum = 0
    for lat, lng in coords:
        lat_decimal = abs(lat) - int(abs(lat))
        lng_decimal = abs(lng) - int(abs(lng))
        
        # Convert to meaningful numbers
        lat_digits = int(lat_decimal * 1000) % 10
        lng_digits = int(lng_decimal * 1000) % 10
        
        decimal_sum += lat_digits + lng_digits
    
    # Return a meaningful number (could be sum mod 10, or just the sum)
    if decimal_sum > 0:
        return str(decimal_sum % 10)
    
    # Fallback: return coordinate count
    return str(len(coords))

# Challenge 3: Cipher decryption functions
def decrypt_rotation_cipher(text: str, shift: int = 13) -> str:
    """Decrypt rotation cipher (ROT cipher)"""
    result = []
    for char in text:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            shifted = (ord(char) - ascii_offset - shift) % 26
            result.append(chr(shifted + ascii_offset))
        else:
            result.append(char)
    return ''.join(result)

def decrypt_railfence(text: str, rails: int = 3) -> str:
    """Decrypt rail fence cipher"""
    if rails <= 1:
        return text
    
    length = len(text)
    fence = [[None for _ in range(length)] for _ in range(rails)]
    
    # Mark the positions
    rail = 0
    direction = 1
    for i in range(length):
        fence[rail][i] = True
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction = -direction
    
    # Fill the fence with cipher text
    index = 0
    for i in range(rails):
        for j in range(length):
            if fence[i][j] and index < length:
                fence[i][j] = text[index]
                index += 1
    
    # Read the plain text
    result = []
    rail = 0
    direction = 1
    for i in range(length):
        result.append(fence[rail][i])
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction = -direction
    
    return ''.join(result)

def decrypt_keyword(text: str, keyword: str = "SHADOW") -> str:
    """Decrypt keyword substitution cipher"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Remove duplicates from keyword and create cipher alphabet
    key = ''.join(dict.fromkeys(keyword.upper()))
    cipher_alphabet = key + ''.join([c for c in alphabet if c not in key])
    
    # Create mapping from cipher to normal alphabet
    mapping = {cipher: normal for cipher, normal in zip(cipher_alphabet, alphabet)}
    
    result = []
    for char in text.upper():
        if char in mapping:
            result.append(mapping[char])
        else:
            result.append(char)
    
    return ''.join(result)

def decrypt_polybius(text: str) -> str:
    """Decrypt Polybius square cipher"""
    # Standard 5x5 Polybius square (I/J combined)
    polybius_square = {
        '11': 'A', '12': 'B', '13': 'C', '14': 'D', '15': 'E',
        '21': 'F', '22': 'G', '23': 'H', '24': 'I', '25': 'K',
        '31': 'L', '32': 'M', '33': 'N', '34': 'O', '35': 'P',
        '41': 'Q', '42': 'R', '43': 'S', '44': 'T', '45': 'U',
        '51': 'V', '52': 'W', '53': 'X', '54': 'Y', '55': 'Z'
    }
    
    result = []
    # Split text into pairs of digits
    for i in range(0, len(text), 2):
        if i + 1 < len(text):
            pair = text[i:i+2]
            if pair in polybius_square:
                result.append(polybius_square[pair])
            else:
                result.append('?')
    
    return ''.join(result)

def parse_and_decrypt_log(log_entry: str) -> str:
    """Parse log entry and decrypt based on cipher type"""
    # Extract cipher type and encrypted payload using regex
    cipher_match = re.search(r'CIPHER_TYPE:\s*(\w+)', log_entry)
    payload_match = re.search(r'ENCRYPTED_PAYLOAD:\s*(\w+)', log_entry)
    
    if not cipher_match or not payload_match:
        return "ERROR: Could not parse log entry"
    
    cipher_type = cipher_match.group(1)
    encrypted_payload = payload_match.group(1)
    
    # Decrypt based on cipher type
    if cipher_type == "ROTATION_CIPHER":
        # Try different rotation values (common ones: 13, 1, 25)
        for shift in [13, 1, 25, 3, 7, 11, 17, 19, 23]:
            result = decrypt_rotation_cipher(encrypted_payload, shift)
            if result.isalpha():  # Check if result makes sense
                return result
        return decrypt_rotation_cipher(encrypted_payload, 13)  # Default ROT13
    elif cipher_type == "RAILFENCE":
        return decrypt_railfence(encrypted_payload, 3)
    elif cipher_type == "KEYWORD":
        return decrypt_keyword(encrypted_payload, "SHADOW")
    elif cipher_type == "POLYBIUS":
        return decrypt_polybius(encrypted_payload)
    else:
        return f"ERROR: Unknown cipher type {cipher_type}"

# Challenge 4: Final message decryption
def decrypt_final_message(param1: str, param2: str, param3: str) -> str:
    """Combine all components for final decryption"""
    # Based on Intel hints:
    # Intel 1: "Their cipher isn't relying on a single safeguard — something else is reinforcing the pattern"
    # Intel 2: "The intercepted number isn't random. It strengthens the lock alongside the keyword, affecting the message as a whole"
    # Intel 3: "The intruders fortified their cipher"
    #
    # This suggests:
    # - param1 (from challenge 1) is the main message to decrypt
    # - param3 (from challenge 3) is the keyword
    # - param2 (from challenge 2) is the number that "strengthens" the cipher
    # - Both the keyword and number work together to affect the whole message
    
    # Method 1: Fortified keyword cipher - use both keyword and number
    if param3.isalpha() and param2.isdigit():
        keyword = param3.upper()
        number = int(param2) % 26
        
        # Create a "fortified" keyword by shifting it with the number
        fortified_keyword = ""
        for char in keyword:
            if char.isalpha():
                shifted = (ord(char) - ord('A') + number) % 26
                fortified_keyword += chr(shifted + ord('A'))
            else:
                fortified_keyword += char
        
        # Use the fortified keyword for decryption
        result = decrypt_keyword(param1, fortified_keyword)
        if result and result.isalpha() and len(result) > 2:
            return result
    
    # Method 2: Double encryption - first keyword, then number shift
    if param3.isalpha() and param2.isdigit():
        # First decrypt with keyword
        intermediate = decrypt_keyword(param1, param3)
        # Then apply rotation cipher with the number
        number = int(param2) % 26
        result = decrypt_rotation_cipher(intermediate, number)
        if result and result.isalpha() and len(result) > 2:
            return result
    
    # Method 3: Number-modified keyword cipher
    if param3.isalpha() and param2.isdigit():
        # Use the number to modify how the keyword cipher works
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        keyword = param3.upper()
        number = int(param2) % 26
        
        # Create cipher alphabet starting from the number position
        rotated_alphabet = alphabet[number:] + alphabet[:number]
        
        # Remove duplicates from keyword and create modified cipher alphabet
        key = ''.join(dict.fromkeys(keyword))
        cipher_alphabet = key + ''.join([c for c in rotated_alphabet if c not in key])
        
        # Create mapping from cipher to normal alphabet
        mapping = {cipher: normal for cipher, normal in zip(cipher_alphabet, alphabet)}
        
        result = []
        for char in param1.upper():
            if char in mapping:
                result.append(mapping[char])
            else:
                result.append(char)
        
        final_result = ''.join(result)
        if final_result and final_result.isalpha() and len(final_result) > 2:
            return final_result
    
    # Method 4: Vigenère-like cipher using both keyword and number
    if param3.isalpha() and param2.isdigit():
        keyword = param3.upper()
        number = int(param2) % 26
        message = param1.upper()
        
        # Extend keyword to match message length
        extended_key = ""
        for i in range(len(message)):
            if message[i].isalpha():
                key_char = keyword[i % len(keyword)]
                # Modify key character by the number
                modified_key_char = chr(((ord(key_char) - ord('A') + number) % 26) + ord('A'))
                extended_key += modified_key_char
            else:
                extended_key += message[i]
        
        # Decrypt using the extended key
        result = []
        for i, char in enumerate(message):
            if char.isalpha():
                key_shift = ord(extended_key[i]) - ord('A')
                decrypted_char = chr(((ord(char) - ord('A') - key_shift) % 26) + ord('A'))
                result.append(decrypted_char)
            else:
                result.append(char)
        
        final_result = ''.join(result)
        if final_result and final_result.isalpha() and len(final_result) > 2:
            return final_result
    
    # Method 5: Simple combination fallbacks
    if param2.isdigit():
        shift = int(param2) % 26
        result = decrypt_rotation_cipher(param1, shift)
        if result and result.isalpha() and len(result) > 2:
            return result
    
    if param3.isalpha():
        result = decrypt_keyword(param1, param3)
        if result and result.isalpha() and len(result) > 2:
            return result
    
    # Final fallback
    return f"{param1}_{param2}_{param3}"

def solve_operation_safeguard(data: dict) -> dict:
    """Main function to solve all challenges"""
    try:
        # Challenge 1: Reverse transformations
        transformations = data['challenge_one']['transformations']
        transformed_word = data['challenge_one']['transformed_encrypted_word']
        
        challenge1_result = reverse_transformations(transformed_word, transformations)
        
        # Challenge 2: Coordinate pattern analysis
        coordinates = data['challenge_two']
        challenge2_result = analyze_coordinates(coordinates)
        
        # Challenge 3: Log decryption
        log_entry = data['challenge_three']
        challenge3_result = parse_and_decrypt_log(log_entry)
        
        # Challenge 4: Final decryption
        challenge4_result = decrypt_final_message(
            challenge1_result, challenge2_result, challenge3_result
        )
        
        return {
            "challenge_one": challenge1_result,
            "challenge_two": challenge2_result,
            "challenge_three": challenge3_result,
            "challenge_four": challenge4_result
        }
        
    except Exception as e:
        return {"error": str(e)}
