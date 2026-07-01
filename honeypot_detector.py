import json

def is_honeypot(candidate) -> bool:
    """
    Returns True if the candidate is deemed a honeypot (e.g., impossible profiles).
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    
    # 1. Skill mismatch honeypot: >15 skills, but average duration is < 2 months
    if len(skills) > 15:
        total_duration = sum(skill.get("duration_months", 0) for skill in skills)
        avg_duration = total_duration / len(skills)
        if avg_duration < 2:
            return True
            
    # 2. Impossible age / experience
    years_exp = profile.get("years_of_experience", 0)
    career_history = candidate.get("career_history", [])
    if years_exp > 40:
        return True
        
    # 3. Pure academics (no company tenure > 0)
    if len(career_history) > 0:
        has_industry = False
        for role in career_history:
            title = role.get("title", "").lower()
            company = role.get("company", "").lower()
            if "university" not in company and "phd" not in title and "student" not in title:
                has_industry = True
        
        if not has_industry and years_exp > 0:
            return True
            
    return False
