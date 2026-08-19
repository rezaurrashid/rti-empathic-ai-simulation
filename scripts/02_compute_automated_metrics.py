import pandas as pd
import numpy as np
from pathlib import Path
from docx import Document
import re
from scipy.stats import entropy
import torch
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONVERSATION PARSER
# ============================================================================

def parse_conversation_docx(docx_path):
    """
    Extract persona, demographics, and conversation turns from docx file.
    Returns dict with: demographics, persona_card, assistant_turns, patient_turns
    """
    doc = Document(docx_path)
    
    content = {
        'demographics': {},
        'persona_card': '',
        'assistant_turns': [],
        'patient_turns': [],
        'full_conversation': []
    }
    
    current_section = None
    temp_text = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        
        if not text:
            continue
            
        # Detect sections
        if 'Demographics' in text:
            current_section = 'demographics'
            continue
        elif 'Persona Card' in text:
            current_section = 'persona'
            continue
        elif 'Conversation Transcript' in text:
            current_section = 'conversation'
            continue
        
        # Parse demographics
        if current_section == 'demographics':
            if 'Race:' in text:
                content['demographics']['race'] = text.split('Race:')[-1].split(',')[0].strip()
            if 'Gender:' in text:
                content['demographics']['gender'] = text.split('Gender:')[-1].split(',')[0].strip()
            if 'Age:' in text:
                content['demographics']['age'] = text.split('Age:')[-1].strip()
        
        # Parse persona card
        elif current_section == 'persona':
            temp_text.append(text)
        
        # Parse conversation turns
        elif current_section == 'conversation':
            if text.startswith('A:') or text.startswith('ASSISTANT:'):
                turn_text = re.sub(r'^(A:|ASSISTANT:)\s*', '', text).strip()
                content['assistant_turns'].append(turn_text)
                content['full_conversation'].append(('assistant', turn_text))
            elif text.startswith('PATIENT:'):
                turn_text = re.sub(r'^PATIENT:\s*', '', text).strip()
                content['patient_turns'].append(turn_text)
                content['full_conversation'].append(('patient', turn_text))
    
    # Join persona text
    content['persona_card'] = ' '.join(temp_text)
    
    return content

# ============================================================================
# 2. RESPONSE-TO-CONTEXT RELEVANCE (NO REFERENCE NEEDED)
# ============================================================================

def calculate_response_relevance(assistant_turns, patient_turns):
    """
    Measure semantic similarity between patient utterances and assistant responses.
    Uses sentence embeddings (no reference needed).
    """
    if len(assistant_turns) == 0 or len(patient_turns) == 0:
        return {'mean_relevance': 0.0, 'std_relevance': 0.0}
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    relevance_scores = []
    for i in range(min(len(assistant_turns), len(patient_turns))):
        # Get embeddings
        patient_emb = model.encode(patient_turns[i], convert_to_tensor=True)
        assistant_emb = model.encode(assistant_turns[i], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            patient_emb.unsqueeze(0), 
            assistant_emb.unsqueeze(0)
        ).item()
        
        relevance_scores.append(similarity)
    
    return {
        'mean_relevance': np.mean(relevance_scores),
        'std_relevance': np.std(relevance_scores)
    }

# ============================================================================
# 3. LEXICAL DIVERSITY (NO REFERENCE NEEDED)
# ============================================================================

def calculate_lexical_diversity(assistant_turns):
    """
    Measure vocabulary richness using Type-Token Ratio (TTR) and related metrics.
    """
    if not assistant_turns:
        return {
            'type_token_ratio': 0.0,
            'unique_words': 0,
            'total_words': 0,
            'avg_word_length': 0.0
        }
    
    # Tokenize all text
    all_words = []
    for turn in assistant_turns:
        words = re.findall(r'\b\w+\b', turn.lower())
        all_words.extend(words)
    
    if not all_words:
        return {
            'type_token_ratio': 0.0,
            'unique_words': 0,
            'total_words': 0,
            'avg_word_length': 0.0
        }
    
    unique_words = set(all_words)
    
    return {
        'type_token_ratio': len(unique_words) / len(all_words),
        'unique_words': len(unique_words),
        'total_words': len(all_words),
        'avg_word_length': np.mean([len(word) for word in all_words])
    }

# ============================================================================
# 4. RESPONSE LENGTH APPROPRIATENESS (NO REFERENCE NEEDED)
# ============================================================================

def calculate_response_length_metrics(assistant_turns):
    """
    Analyze response length patterns and appropriateness.
    """
    if not assistant_turns:
        return {
            'avg_response_length_words': 0.0,
            'std_response_length': 0.0,
            'max_response_length': 0,
            'min_response_length': 0,
            'sentences_per_response': 0.0
        }
    
    word_counts = [len(turn.split()) for turn in assistant_turns]
    sentence_counts = [len(re.split(r'[.!?]+', turn)) - 1 for turn in assistant_turns]
    
    return {
        'avg_response_length_words': np.mean(word_counts),
        'std_response_length': np.std(word_counts),
        'max_response_length': max(word_counts),
        'min_response_length': min(word_counts),
        'sentences_per_response': np.mean(sentence_counts)
    }

# ============================================================================
# 5. SDOH RELEVANCE SCORING (NO REFERENCE NEEDED)
# ============================================================================

def calculate_sdoh_relevance(assistant_turns):
    """
    Calculate proportion of assistant turns that address SDoH barriers.
    """
    sdoh_categories = {
        'transportation': ['ride', 'transport', 'drive', 'car', 'bus', 'shuttle', 'gas', 'parking', 'trip'],
        'financial': ['cost', 'pay', 'bill', 'afford', 'money', 'insurance', 'copay', 'fee', 'financial', 'expense', 'voucher', 'discount', 'price'],
        'housing': ['home', 'house', 'apartment', 'living', 'utilities', 'power', 'electric', 'rent', 'housing'],
        'employment': ['work', 'job', 'employer', 'shift', 'hours', 'leave', 'employment', 'boss', 'career'],
        'food': ['food', 'meal', 'eat', 'nutrition', 'hungry', 'groceries', 'cook'],
        'social_support': ['family', 'caregiver', 'support', 'help', 'alone', 'isolated', 'friend', 'neighbor', 'church'],
        'language': ['language', 'translate', 'understand', 'read', 'forms', 'instructions', 'english'],
        'technology': ['portal', 'app', 'phone', 'computer', 'online', 'digital', 'internet', 'email']
    }
    
    if not assistant_turns:
        return {
            'sdoh_overall_relevance': 0.0,
            'sdoh_transportation': 0.0,
            'sdoh_financial': 0.0,
            'sdoh_housing': 0.0,
            'sdoh_employment': 0.0,
            'sdoh_food': 0.0,
            'sdoh_social_support': 0.0,
            'sdoh_language': 0.0,
            'sdoh_technology': 0.0
        }
    
    category_scores = {}
    overall_count = 0
    
    for category, keywords in sdoh_categories.items():
        category_count = 0
        for turn in assistant_turns:
            turn_lower = turn.lower()
            if any(keyword in turn_lower for keyword in keywords):
                category_count += 1
        
        category_scores[f'sdoh_{category}'] = category_count / len(assistant_turns)
        if category_count > 0:
            overall_count += 1
    
    # Overall: proportion of turns mentioning ANY SDoH
    sdoh_turns = sum(1 for turn in assistant_turns 
                     if any(any(kw in turn.lower() for kw in keywords) 
                           for keywords in sdoh_categories.values()))
    
    category_scores['sdoh_overall_relevance'] = sdoh_turns / len(assistant_turns)
    
    return category_scores

# ============================================================================
# 6. EMPATHY CUE USAGE (NO REFERENCE NEEDED)
# ============================================================================

def calculate_empathy_cues(assistant_turns):
    """
    Calculate proportion of turns with explicit empathic language.
    """
    empathy_categories = {
        'acknowledgment': ['i hear', 'i understand', 'i can see', 'i know', 'i realize'],
        'validation': ['sounds like', 'seems like', 'makes sense', 'understandable', 'reasonable'],
        'appreciation': ['appreciate', 'thank', 'grateful', 'impressed', 'respect'],
        'difficulty': ['frustrating', 'difficult', 'hard', 'tough', 'exhausting', 'challenging'],
        'emotional_support': ['no wonder', 'must be', 'can imagine', 'feel', 'emotion']
    }
    
    if not assistant_turns:
        return {
            'empathy_overall': 0.0,
            'empathy_acknowledgment': 0.0,
            'empathy_validation': 0.0,
            'empathy_appreciation': 0.0,
            'empathy_difficulty': 0.0,
            'empathy_support': 0.0
        }
    
    category_scores = {}
    
    for category, phrases in empathy_categories.items():
        category_count = 0
        for turn in assistant_turns:
            turn_lower = turn.lower()
            if any(phrase in turn_lower for phrase in phrases):
                category_count += 1
        category_scores[f'empathy_{category}'] = category_count / len(assistant_turns)
    
    # Overall empathy
    empathy_turns = sum(1 for turn in assistant_turns 
                       if any(any(phrase in turn.lower() for phrase in phrases) 
                             for phrases in empathy_categories.values()))
    
    category_scores['empathy_overall'] = empathy_turns / len(assistant_turns)
    
    return category_scores

# ============================================================================
# 7. QUESTION ASKING PATTERN (NO REFERENCE NEEDED)
# ============================================================================

def calculate_question_patterns(assistant_turns):
    """
    Analyze types and frequency of questions asked.
    """
    if not assistant_turns:
        return {
            'question_ratio': 0.0,
            'open_ended_ratio': 0.0,
            'closed_ended_ratio': 0.0,
            'avg_questions_per_turn': 0.0
        }
    
    total_questions = 0
    open_ended = 0
    closed_ended = 0
    
    open_starters = ['what', 'how', 'why', 'tell me', 'describe', 'explain', 'share']
    closed_starters = ['do you', 'did you', 'can you', 'will you', 'are you', 'is it', 'would you']
    
    for turn in assistant_turns:
        questions = [s.strip() for s in turn.split('?') if s.strip()]
        total_questions += len(questions)
        
        for q in questions:
            q_lower = q.lower()
            if any(q_lower.startswith(starter) or starter in q_lower[:30] for starter in open_starters):
                open_ended += 1
            elif any(q_lower.startswith(starter) or starter in q_lower[:30] for starter in closed_starters):
                closed_ended += 1
    
    return {
        'question_ratio': total_questions / len(assistant_turns),
        'open_ended_ratio': open_ended / total_questions if total_questions > 0 else 0.0,
        'closed_ended_ratio': closed_ended / total_questions if total_questions > 0 else 0.0,
        'avg_questions_per_turn': total_questions / len(assistant_turns)
    }

# ============================================================================
# 8. CONTEXTUAL COHERENCE (NO REFERENCE NEEDED)
# ============================================================================

def calculate_contextual_coherence(full_conversation):
    """
    Measure how consistent topics are across conversation turns.
    """
    if len(full_conversation) < 2:
        return {'contextual_coherence': 0.0}
    
    def get_keywords(text):
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was',
                      'you', 'i', 'me', 'my', 'your', 'it', 'that', 'this'}
        words = set(re.findall(r'\b\w+\b', text.lower()))
        return words - stop_words
    
    coherence_scores = []
    for i in range(len(full_conversation) - 1):
        curr_keywords = get_keywords(full_conversation[i][1])
        next_keywords = get_keywords(full_conversation[i+1][1])
        
        if curr_keywords and next_keywords:
            overlap = len(curr_keywords & next_keywords)
            union = len(curr_keywords | next_keywords)
            coherence_scores.append(overlap / union if union > 0 else 0)
    
    return {
        'contextual_coherence': np.mean(coherence_scores) if coherence_scores else 0.0
    }

# ============================================================================
# 9. EMOTIONAL ENTROPY (NO REFERENCE NEEDED)
# ============================================================================

def calculate_emotional_entropy(assistant_turns):
    """
    Measure whether emotional tone remains stable throughout dialogue.
    """
    if not assistant_turns:
        return {'emotional_entropy': 0.0, 'emotional_consistency': 0.0}
    
    emotion_indicators = {
        'supportive': ['help', 'support', 'here for', 'assist', 'guide'],
        'empathic': ['understand', 'hear', 'sounds', 'difficult', 'hard'],
        'directive': ['you should', 'you need', 'must', 'have to', 'will'],
        'questioning': ['what', 'how', 'when', 'where', 'why', 'tell me'],
        'acknowledging': ['appreciate', 'thank', 'good', 'great', 'impressed']
    }
    
    emotion_counts = {emotion: 0 for emotion in emotion_indicators}
    
    for turn in assistant_turns:
        turn_lower = turn.lower()
        for emotion, keywords in emotion_indicators.items():
            if any(keyword in turn_lower for keyword in keywords):
                emotion_counts[emotion] += 1
    
    total = sum(emotion_counts.values())
    if total == 0:
        return {'emotional_entropy': 0.0, 'emotional_consistency': 1.0}
    
    probabilities = [count / total for count in emotion_counts.values() if count > 0]
    ent = entropy(probabilities) if len(probabilities) > 1 else 0.0
    
    # Consistency is inverse of entropy (normalized)
    max_entropy = np.log(len(emotion_indicators))
    consistency = 1 - (ent / max_entropy) if max_entropy > 0 else 1.0
    
    return {
        'emotional_entropy': ent,
        'emotional_consistency': consistency
    }

# ============================================================================
# 10. ACTION-ORIENTATION (NO REFERENCE NEEDED)
# ============================================================================

def calculate_action_orientation(assistant_turns):
    """
    Measure how often assistant provides concrete action steps.
    """
    if not assistant_turns:
        return {
            'action_ratio': 0.0,
            'concrete_suggestions': 0.0,
            'information_provision': 0.0
        }
    
    action_phrases = [
        'i can', 'i will', 'let me', "i'll", 'we can', 'you can',
        'call', 'contact', 'schedule', 'arrange', 'set up',
        'here is', "here's", 'try', 'ask for'
    ]
    
    concrete_phrases = [
        'number is', 'address is', 'at ', 'on ', 'step ', 'first', 'next',
        'today', 'tomorrow', 'this week', 'specific', 'exactly'
    ]
    
    info_phrases = [
        'usually', 'typically', 'often', 'sometimes', 'most', 'common',
        'generally', 'normal', 'expected'
    ]
    
    action_count = 0
    concrete_count = 0
    info_count = 0
    
    for turn in assistant_turns:
        turn_lower = turn.lower()
        if any(phrase in turn_lower for phrase in action_phrases):
            action_count += 1
        if any(phrase in turn_lower for phrase in concrete_phrases):
            concrete_count += 1
        if any(phrase in turn_lower for phrase in info_phrases):
            info_count += 1
    
    return {
        'action_ratio': action_count / len(assistant_turns),
        'concrete_suggestions': concrete_count / len(assistant_turns),
        'information_provision': info_count / len(assistant_turns)
    }

# ============================================================================
# 11. CONVERSATION LENGTH METRICS
# ============================================================================

def calculate_length_metrics(assistant_turns, patient_turns):
    """
    Calculate basic conversation length statistics.
    """
    return {
        'total_turns': len(assistant_turns) + len(patient_turns),
        'assistant_turns': len(assistant_turns),
        'patient_turns': len(patient_turns),
        'avg_assistant_words': np.mean([len(turn.split()) for turn in assistant_turns]) if assistant_turns else 0,
        'avg_patient_words': np.mean([len(turn.split()) for turn in patient_turns]) if patient_turns else 0
    }

# ============================================================================
# 12. MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_single_conversation(docx_path):
    """
    Evaluate a single conversation file - NO REFERENCE TEXT NEEDED.
    """
    print(f"Evaluating: {docx_path.name}")
    
    # Parse conversation
    conv = parse_conversation_docx(docx_path)
    
    # Calculate all metrics
    results = {
        'conversation_id': docx_path.stem,
        'race': conv['demographics'].get('race', 'Unknown'),
        'gender': conv['demographics'].get('gender', 'Unknown'),
        'age': conv['demographics'].get('age', 'Unknown'),
    }
    
    # Response-to-Context Relevance (replaces BERTScore)
    relevance = calculate_response_relevance(conv['assistant_turns'], conv['patient_turns'])
    results.update(relevance)
    
    # Lexical Diversity (replaces BLEU)
    diversity = calculate_lexical_diversity(conv['assistant_turns'])
    results.update(diversity)
    
    # Response Length Appropriateness (replaces ROUGE)
    length_metrics = calculate_response_length_metrics(conv['assistant_turns'])
    results.update(length_metrics)
    
    # SDoH Relevance (detailed breakdown)
    sdoh = calculate_sdoh_relevance(conv['assistant_turns'])
    results.update(sdoh)
    
    # Empathy Cues (detailed breakdown)
    empathy = calculate_empathy_cues(conv['assistant_turns'])
    results.update(empathy)
    
    # Question Patterns
    questions = calculate_question_patterns(conv['assistant_turns'])
    results.update(questions)
    
    # Contextual Coherence
    coherence = calculate_contextual_coherence(conv['full_conversation'])
    results.update(coherence)
    
    # Emotional Entropy
    emotion = calculate_emotional_entropy(conv['assistant_turns'])
    results.update(emotion)
    
    # Action Orientation
    action = calculate_action_orientation(conv['assistant_turns'])
    results.update(action)
    
    # Basic length metrics
    basic_length = calculate_length_metrics(conv['assistant_turns'], conv['patient_turns'])
    results.update(basic_length)
    
    return results

def evaluate_all_conversations(conversation_dir, output_csv):
    """
    Evaluate all conversation files - NO REFERENCE TEXT NEEDED.
    """
    conv_dir = Path(conversation_dir)
    docx_files = sorted(list(conv_dir.glob('*.docx')))
    
    if not docx_files:
        print(f"No .docx files found in {conversation_dir}")
        return
    
    print(f"Found {len(docx_files)} conversation files to evaluate")
    print("=" * 70)
    print("NOTE: Using reference-free metrics - no gold standard needed!")
    print("=" * 70)
    
    all_results = []
    
    for i, docx_path in enumerate(docx_files, 1):
        try:
            results = evaluate_single_conversation(docx_path)
            all_results.append(results)
            print(f"✓ Completed {i}/{len(docx_files)}: {docx_path.name}")
        except Exception as e:
            print(f"✗ Error processing {docx_path.name}: {str(e)}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    
    print("=" * 70)
    print(f"Evaluation complete! Results saved to: {output_csv}")
    print(f"Successfully evaluated: {len(all_results)}/{len(docx_files)} conversations")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Key metrics summary
    key_metrics = [
        'mean_relevance', 'type_token_ratio', 'sdoh_overall_relevance',
        'empathy_overall', 'contextual_coherence', 'action_ratio'
    ]
    
    if all(col in df.columns for col in key_metrics):
        summary = df[key_metrics].describe()
        print(summary)
    
    return df

# ============================================================================
# 13. USAGE EXAMPLE
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute reference-free automated dialogue metrics for simulated conversations."
    )
    parser.add_argument(
        "--conversations",
        default="outputs/conversations",
        help="Directory containing conv_*.docx files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluation/automated_metrics.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = evaluate_all_conversations(args.conversations, output_path)

    if results_df is not None:
        key_metrics = [
            "mean_relevance",
            "type_token_ratio",
            "sdoh_overall_relevance",
            "empathy_overall",
            "contextual_coherence",
            "action_ratio",
        ]
        available = [c for c in key_metrics if c in results_df.columns]
        if available:
            print("\nKey metric means:")
            print(results_df[available].mean().round(3))
