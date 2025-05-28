import csv
import random
import re
from heapq import nlargest
from rapidfuzz import fuzz
from collections import defaultdict

random.seed(42)

def extract_ocr_words(ocr_text):
    """
    Extracts just the words from OCR text that includes coordinates.
    Each line in OCR text is formatted as: "word, (x1, y1), ..."
    """
    words = []
    for line in ocr_text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split(',', 1)
        if parts and parts[0].strip():
            words.append(parts[0].strip().lower())
    return words


def remove_noise_words(words, noise_words, threshold=0.8):
    """
    Removes tokens that approximately match any noise word.
    Returns both cleaned words and a list of removed words with their matches.
    """
    cleaned = []
    removed = []
    
    for w in words:
        is_noise = False
        for n in noise_words:
            ratio = fuzz.ratio(w, n) / 100.0
            if ratio > threshold:
                removed.append((w, n, ratio))
                is_noise = True
                break
        if not is_noise:
            cleaned.append(w)
    
    return cleaned, removed


def fuzzy_token_set_ratio(name, words):
    """
    Calculate token set ratio and return match details
    """
    tokens = name.lower().split()
    if not tokens:
        return 0, []
    
    matches = []
    total = 0
    
    for t in tokens:
        best_score = 0
        best_word = None
        
        for w in words:
            score = fuzz.ratio(t, w) / 100.0
            if score > best_score:
                best_score = score
                best_word = w
        
        total += best_score
        matches.append((t, best_word, best_score))
    
    return total / len(tokens), matches


def full_name_concatenation_score(candidate, words):
    """
    Checks fuzzy-similarity of full name (no spaces) against each OCR token.
    Returns score and best match details.
    """
    name = candidate[0].lower().replace(" ", "")
    if len(name) < 6:
        return 0, None
    
    best = 0
    best_match = None
    
    for w in words:
        if w == name:
            return 1.0, (w, 1.0)
        
        score = fuzz.ratio(w, name) / 100.0
        if score > best:
            best = score
            best_match = (w, score)
    
    return best, best_match


def score_candidate(candidate, words, threshold=0.8):
    """
    Score a candidate name and return detailed breakdown of matching components
    """
    name = candidate[0]
    tokens = name.lower().split()
    if not tokens:
        return 0, {}
    
    results = {}
    
    # Full concatenation check
    full_score, full_match = full_name_concatenation_score(candidate, words)
    results['full_concatenation'] = {
        'score': full_score,
        'match': full_match
    }
    
    # First & last token matching
    first, last = tokens[0], tokens[-1]
    
    first_best_score = 0
    first_best_word = None
    for w in words:
        score = fuzz.ratio(first, w) / 100.0
        if score > first_best_score:
            first_best_score = score
            first_best_word = w
    
    last_best_score = 0
    last_best_word = None
    for w in words:
        score = fuzz.ratio(last, w) / 100.0
        if score > last_best_score:
            last_best_score = score
            last_best_word = w
    
    results['first_token'] = {
        'token': first,
        'best_match': first_best_word,
        'score': first_best_score
    }
    
    results['last_token'] = {
        'token': last,
        'best_match': last_best_word,
        'score': last_best_score
    }
    
    if first_best_score > threshold and last_best_score > threshold:
        base = 1.0
    elif first_best_score > threshold or last_best_score > threshold:
        base = 0.7
    else:
        base = 0.0
    
    # Bonus for middle tokens
    bonus = 0
    middle_matches = []
    
    if len(tokens) > 2:
        mids = tokens[1:-1]
        for mid in mids:
            best_score = 0
            best_word = None
            for w in words:
                score = fuzz.ratio(mid, w) / 100.0
                if score > best_score:
                    best_score = score
                    best_word = w
            middle_matches.append({
                'token': mid,
                'best_match': best_word,
                'score': best_score
            })
        
        ratios = [m['score'] for m in middle_matches]
        bonus = sum(r for r in ratios if r > threshold) / len(mids) * 0.3
    
    results['middle_tokens'] = middle_matches
    first_last = min(1.0, base + bonus)
    results['first_last_score'] = first_last
    
    # Token set ratio
    set_ratio, set_matches = fuzzy_token_set_ratio(name, words)
    results['token_set_ratio'] = {
        'score': set_ratio,
        'matches': set_matches
    }
    
    # All-parts match
    all_parts_matches = []
    for t in tokens:
        best_score = 0
        best_word = None
        for w in words:
            score = fuzz.ratio(t, w) / 100.0
            if score > best_score:
                best_score = score
                best_word = w
        
        all_parts_matches.append({
            'token': t,
            'best_match': best_word,
            'score': best_score,
            'matched': best_score > threshold
        })
    
    matched = sum(1 for match in all_parts_matches if match['matched'])
    all_parts = matched / len(tokens)
    results['all_parts_match'] = {
        'score': all_parts,
        'details': all_parts_matches
    }
    
    # Token frequency
    freq_matches = []
    for t in tokens:
        exact_matches = [w for w in words if t == w]
        freq_matches.append({
            'token': t,
            'exact_matches': exact_matches,
            'count': len(exact_matches)
        })
    
    freq = sum(match['count'] for match in freq_matches)
    token_freq = freq / max(1, len(tokens))
    results['token_frequency'] = {
        'score': token_freq,
        'details': freq_matches
    }
    
    # Adaptive weighting
    if full_score > 0.85:
        # concatenation-mode
        weights = {
            'first_last': 0.10,
            'set_ratio': 0.25,
            'all_parts': 0.10,
            'token_freq': 0.05,
            'full_score': 0.50
        }
        final = (
            first_last  * weights['first_last'] +
            set_ratio   * weights['set_ratio'] +
            all_parts   * weights['all_parts'] +
            token_freq  * weights['token_freq'] +
            full_score  * weights['full_score']
        )
    else:
        # traditional
        weights = {
            'first_last': 0.15,
            'set_ratio': 0.40,
            'all_parts': 0.20,
            'token_freq': 0.25,
            'full_score': 0.00
        }
        final = (
            first_last  * weights['first_last'] +
            set_ratio   * weights['set_ratio'] +
            all_parts   * weights['all_parts'] +
            token_freq  * weights['token_freq']
        )
    
    results['weights'] = weights
    results['final_score'] = final
    
    return final, results


def approximate_receiver_name(ocr_text, candidate_names, noise_words,
                              threshold=0.8, match_threshold=0.36, explain=False):
    """
    Find the best matching candidate name in OCR text with detailed explanations
    """
    words = extract_ocr_words(ocr_text)
    cleaned, removed_noise = remove_noise_words(words, noise_words, threshold)
    
    print(f"Original OCR word count: {len(words)}")
    print(f"After noise removal: {len(cleaned)} words")
    
    scores_with_explanations = []
    for cand in candidate_names:
        score, details = score_candidate(cand, cleaned, threshold)
        scores_with_explanations.append((cand, score, details))
    
    top = nlargest(5, scores_with_explanations, key=lambda x: x[1])
    
    # Print the top 5 candidates with their scores and explanations
    print("\nTop 5 candidates:")
    for i, (candidate, score, details) in enumerate(top, 1):
        print(f"\n{i}. {candidate[0]} (Score: {score:.4f})")
        
        if explain:
            print_candidate_explanation(candidate, details, cleaned)
    
    if top and top[0][1] >= match_threshold:
        return top[0][0], top[0][1], top[0][2]
    return None, 0, {}


def print_candidate_explanation(candidate, details, cleaned_words):
    """
    Print detailed explanation of why a candidate received its score
    """
    name = candidate[0]
    tokens = name.lower().split()
    
    print(f"  Explanation for '{name}':")
    
    # Full name concatenation
    if details['full_concatenation']['score'] > 0:
        concat_name = name.lower().replace(" ", "")
        match_word, match_score = details['full_concatenation']['match']
        print(f"  - Full name '{concat_name}' matches OCR word '{match_word}' with {match_score:.2f} similarity")
    
    # First and last tokens
    print(f"  - First token '{details['first_token']['token']}' best matches '{details['first_token']['best_match']}' ({details['first_token']['score']:.2f})")
    print(f"  - Last token '{details['last_token']['token']}' best matches '{details['last_token']['best_match']}' ({details['last_token']['score']:.2f})")
    
    # Middle tokens if any
    if details['middle_tokens']:
        print(f"  - Middle tokens:")
        for mid in details['middle_tokens']:
            print(f"    - '{mid['token']}' best matches '{mid['best_match']}' ({mid['score']:.2f})")
    
    # Token set ratio
    print(f"  - Token set ratio: {details['token_set_ratio']['score']:.2f}")
    
    # All parts match
    print(f"  - All parts match score: {details['all_parts_match']['score']:.2f}")
    for part in details['all_parts_match']['details']:
        status = "✓" if part['matched'] else "✗"
        print(f"    - {status} '{part['token']}' matches '{part['best_match']}' ({part['score']:.2f})")
    
    # Token frequency
    print(f"  - Token frequency score: {details['token_frequency']['score']:.2f}")
    for freq in details['token_frequency']['details']:
        if freq['count'] > 0:
            print(f"    - '{freq['token']}' appears exactly {freq['count']} times in OCR text")
    
    # Weights applied
    print(f"  - Weights applied: {details['weights']}")
    if details['weights']['full_score'] > 0:
        print(f"    (Using concatenation-mode weighting due to high full name match)")
    else:
        print(f"    (Using traditional token-based weighting)")
    
    print(f"  - Final score: {details['final_score']:.4f}")

# List of known noise words (addresses, numbers, etc.)
noise_words = [
    "ni","no","resi","calvin",
    "institute","technology","menara","lt","rmci","jl","industri","blok",
    "b14","kav","kemayoran","pusat","10610","pademangan","kota","administrasi","utara",
    "dki","14410","jakarta","barat","express","tower","tangerang","jalan","alamat",
    "timur","spx","cit","penerima","pengirim","shopee","sameday","jkt","berat","ongkir","kg",
    "reguler",'tokopedia', 'sicepat', 'reg', 'sto', 'std', 'sp', 'central', 'pesanan', 'variasi', 'sku', 'produk',
    "rw", 'east', 'cod', 'cek', 'dulu', 'tidak', 'batas', 'kirim', "biok", "str", 'anteraja', 'oleh', 'murah',
    "shipping", "label", "gtl", "cllining", "wabehause", "class", "taaaattatatata", "drigin", "sorting",
    "tahcsit", "cit", "incino", "vanause", "it", "ship", "to", "address", "ip", "tracking", "numbelt",
    "intusosoaimpua", "ybil", "numbel", "fdt"
]