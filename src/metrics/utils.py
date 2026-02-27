def format_plan_json(idea):
    """Formats a NovelIdea object or dictionary into a string."""
    if hasattr(idea, 'dict'):
        idea = idea.dict()
    
    title = idea.get('title', 'N/A')
    description = idea.get('description', 'N/A')
    reasoning = idea.get('reasoning', 'N/A')
    
    return (
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Reasoning: {reasoning}"
    )

def format_idea_with_abstract(idea):
    """Formats an idea with its title and abstract into a string."""
    if hasattr(idea, 'dict'):
        idea = idea.dict()
    
    title = idea.get('title', 'N/A')
    abstract = idea.get('abstract', 'N/A')
    
    return (
        f"Title: {title}\n"
        f"Abstract: {abstract}"
    )