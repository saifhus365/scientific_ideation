from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Timeline:
    start_date: Optional[str]
    end_date: Optional[str]
    specific_year: Optional[int]

@dataclass
class QueryAnalysis:
    query: str  # Add this line
    topics: List[str]
    timeline: Timeline
    intention: str

class ResponseParser:
    def parse_response(self, text: str, original_query: str) -> QueryAnalysis:
        """Parse JSON response from LLM into QueryAnalysis object"""
        try:
            # Clean the response text
            text = text.strip()
            
            # Find the JSON object
            start = text.find('{')
            end = text.rfind('}')
            
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in response")
                
            json_str = text[start:end + 1]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required fields
            if not all(k in data for k in ['topics', 'timeline', 'intention']):
                raise ValueError("Missing required fields in JSON response")
                
            # Create Timeline object
            timeline_data = data['timeline']
            timeline = Timeline(
                start_date=timeline_data.get('start_date'),
                end_date=timeline_data.get('end_date'),
                specific_year=timeline_data.get('specific_year')
            )
            
            return QueryAnalysis(
                query=original_query,
                topics=data['topics'],
                timeline=timeline,
                intention=data['intention']
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing response: {str(e)}")