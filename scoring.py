def calculate_heuristic_score(candidate_features, semantic_sim):
    """
    Calculates the heuristic score based on semantic similarity and behavioral signals.
    """
    # Normalized features from candidate_features
    response_rate = candidate_features.get('recruiter_response_rate', 0.0)
    github_score_norm = candidate_features.get('github_activity_score', 0) / 100.0 if candidate_features.get('github_activity_score', -1) != -1 else 0.0
    
    # Recency score (inverse of last_active_days, normalized 0-1)
    last_active_days = candidate_features.get('last_active_days', 365)
    recency_score = max(0.0, 1.0 - (last_active_days / 365.0))
    
    # Experience fit score (peaks at 7 years, drops at edges)
    years_exp = candidate_features.get('years_of_experience', 0)
    if 5 <= years_exp <= 9:
        exp_fit_score = 1.0
    else:
        exp_fit_score = max(0.0, 1.0 - abs(years_exp - 7) / 7.0)
        
    # Skill quality ratio (duration / count)
    skill_quality_ratio = candidate_features.get('skill_quality_ratio', 0.0)
    # Normalize to roughly 0-1 (e.g. 24 months / 5 skills = 4.8, cap at 5)
    skill_quality_ratio = min(1.0, skill_quality_ratio / 5.0)

    # Weights based on Claude's recommendation
    score = (semantic_sim * 0.4) + \
            (response_rate * 0.15) + \
            (github_score_norm * 0.1) + \
            (recency_score * 0.1) + \
            (exp_fit_score * 0.15) + \
            (skill_quality_ratio * 0.1)
            
    return score

def ensemble_score(llm_score, heuristic_score):
    """
    Combines LLM score and heuristic score.
    """
    return 0.65 * llm_score + 0.35 * heuristic_score
