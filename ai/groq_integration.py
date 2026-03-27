"""
Groq LLM Integration Module

Handles communication with Groq API for generating natural language responses.
Uses Groq's fast inference with Mixtral 8x7B model for origami-related queries.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")


class GroqClient:
    """Client for interacting with Groq API."""

    def __init__(self):
        """Initialize Groq client with API key from environment."""
        if not GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY not found in environment. "
                "Please add it to your .env file."
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
        self.is_available_flag = True

    def is_available(self) -> bool:
        """Check if Groq API is available and configured."""
        return self.is_available_flag and GROQ_API_KEY is not None

    def generate_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 500
    ) -> Optional[str]:
        """
        Generate a response using Groq API.

        Args:
            prompt: The input prompt for the model.
            temperature: Sampling temperature (0.0-2.0). Default 0.7.
            max_tokens: Maximum tokens in response. Default 500.

        Returns:
            Generated text response, or None if generation fails.
        """
        try:
            if not self.is_available():
                print("⚠️  Groq API not available")
                return None

            message = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return message.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Groq API error: {str(e)}")
            return None


def get_groq_client() -> GroqClient:
    """Get or create Groq client instance."""
    try:
        return GroqClient()
    except ValueError as e:
        print(f"⚠️  {str(e)}")
        return None


def format_search_response_prompt(query: str, results: list) -> str:
    """
    Format a prompt for generating search response.

    Args:
        query: The original user query.
        results: List of database results (tuples of origami data).

    Returns:
        Formatted prompt for Groq.
    """
    results_text = "\n".join(
        [
            f"- {row[0]} by {row[1]} (Difficulty: {row[2] if len(row) > 2 else 'N/A'})"
            for row in results[:5]
        ]
    )

    prompt = f"""Based on the user's query about origami, provide a friendly, helpful response 
using the following database results:

User Query: {query}

Database Results:
{results_text}

Please provide a conversational response (2-3 sentences) that:
1. Directly addresses the query
2. Mentions specific origami models found
3. Includes interesting details about difficulty or creator if available
4. Uses emojis sparingly for visual appeal

Keep response concise and friendly."""
    return prompt


def format_image_response_prompt(
    model_name: str, confidence: float, top_models: list
) -> str:
    """
    Format a prompt for generating image analysis response.

    Args:
        model_name: Predicted origami model name.
        confidence: Prediction confidence (0-1).
        top_models: List of top-3 predictions with scores.

    Returns:
        Formatted prompt for Groq.
    """
    alternatives = "\n".join(
        [f"- {name} (confidence: {score:.1%})" for name, score in top_models[1:3]]
    )

    prompt = f"""Analyze this origami image prediction and provide a brief, friendly response.

Prediction: {model_name}
Confidence: {confidence:.1%}
Other possibilities:
{alternatives}

Please provide a response (2-3 sentences) that:
1. Confirms or expresses confidence in the prediction
2. Mentions confidence level in a natural way
3. References the alternatives if confidence is moderate
4. Adds one interesting fact about the origami model if you know it
5. Is conversational and encouraging

Keep it brief and use emojis if appropriate for origami-related observations."""
    return prompt


def generate_search_response(query: str, results: list) -> Optional[str]:
    """
    Generate a natural language response for a database search.

    Args:
        query: The original user query.
        results: List of database results.

    Returns:
        Generated response text, or None if generation fails.
    """
    client = get_groq_client()
    if not client or not client.is_available():
        return None

    prompt = format_search_response_prompt(query, results)
    return client.generate_response(prompt, temperature=0.7, max_tokens=300)


def generate_image_response(
    model_name: str, confidence: float, top_models: list
) -> Optional[str]:
    """
    Generate a natural language response for an image prediction.

    Args:
        model_name: Predicted origami model name.
        confidence: Prediction confidence (0-1).
        top_models: List of (name, score) tuples of top predictions.

    Returns:
        Generated response text, or None if generation fails.
    """
    client = get_groq_client()
    if not client or not client.is_available():
        return None

    prompt = format_image_response_prompt(model_name, confidence, top_models)
    return client.generate_response(prompt, temperature=0.7, max_tokens=250)


def _format_difficulty_emoji(difficulty: str) -> str:
    """
    Format difficulty level with visual emoji rating.
    
    Args:
        difficulty: Difficulty level text (Beginner/Intermediate/Advanced or numeric 1-5)
        
    Returns:
        Formatted difficulty with emoji stars
    """
    difficulty_lower = str(difficulty).lower()
    
    # Handle numeric difficulty (1-5)
    try:
        if difficulty_lower.isdigit() or (difficulty_lower[0].isdigit()):
            level = int(difficulty_lower[0])
            stars = "⭐" * level + "☆" * (5 - level)
            return f"{stars} ({level}/5)"
    except (ValueError, IndexError):
        pass
    
    # Handle text difficulty
    if "beginner" in difficulty_lower or "easy" in difficulty_lower or difficulty_lower == "1":
        return "🟢 Beginner ⭐ (1/5)"
    elif "intermediate" in difficulty_lower or difficulty_lower == "2" or difficulty_lower == "3":
        return "🟡 Intermediate ⭐⭐ (2-3/5)"
    elif "advanced" in difficulty_lower or "expert" in difficulty_lower or difficulty_lower == "4" or difficulty_lower == "5":
        return "🔴 Advanced ⭐⭐⭐⭐ (4-5/5)"
    
    return f"🟡 {difficulty}"


def _is_valid_tutorial_link(link: str) -> bool:
    """
    Check if tutorial link is valid and not empty.
    
    Args:
        link: Tutorial link URL
        
    Returns:
        True if link is valid, False otherwise
    """
    if not link or not isinstance(link, str):
        return False
    link = link.strip()
    return len(link) > 10 and ("http" in link or "www" in link or ".com" in link)


def generate_professional_image_analysis(
    model_name: str, 
    confidence: float, 
    top_models: list,
    creator: str = "Unknown",
    difficulty: str = "Intermediate",
    paper_shape: str = "Square",
    uses_cutting: bool = False,
    uses_glue: bool = False,
    tutorial_link: str = ""
) -> Optional[str]:
    """
    Generate a professional Origami AI Expert analysis with Markdown formatting.
    
    Args:
        model_name: Predicted origami model name
        confidence: Prediction confidence (0-100)
        top_models: List of (name, score) tuples for top-3
        creator: Model creator name
        difficulty: Difficulty level (Beginner/Intermediate/Advanced or 1-5)
        paper_shape: Paper shape (Square, Rectangle, etc)
        uses_cutting: Whether cutting is required
        uses_glue: Whether glue is required
        tutorial_link: URL to tutorial
        
    Returns:
        Professional Markdown formatted response
    """
    client = get_groq_client()
    if not client or not client.is_available():
        return None
    
    # Format difficulty with emoji
    formatted_difficulty = _format_difficulty_emoji(difficulty)
    
    # Validate and format tutorial link
    tutorial_display = ""
    if _is_valid_tutorial_link(tutorial_link):
        tutorial_display = f"[📚 Tutorial Link]({tutorial_link})"
    else:
        tutorial_display = "📚 Tutorial not available in database"
    
    # Confidence color coding text
    if confidence >= 75:
        confidence_status = "✅ **HIGH CONFIDENCE** - Strong prediction"
    elif confidence >= 50:
        confidence_status = "🟡 **MODERATE CONFIDENCE** - Consider alternatives"
    else:
        confidence_status = "⚠️ **LOW CONFIDENCE** - Use with caution"
    
    # Format top-3 for display
    top3_text = "\n".join([
        f"  {i+1}. **{name}** ({score:.1f}%)" 
        for i, (name, score) in enumerate(top_models[:3])
    ])
    
    prompt = f"""You are a Professional Origami AI Expert Assistant. Generate a detailed analysis in Markdown format.

DETECTED MODEL: {model_name}
CONFIDENCE: {confidence:.1f}% ({confidence_status})

TOP-3 PREDICTIONS:
{top3_text}

DATABASE INFO:
- Creator: {creator}
- Difficulty: {formatted_difficulty}
- Paper Shape: {paper_shape}
- Requires Cutting: {"✅ Yes" if uses_cutting else "❌ No"}
- Requires Glue: {"✅ Yes" if uses_glue else "❌ No"}
- Tutorial: {tutorial_display}

INSTRUCTIONS - Generate output with EXACTLY these sections in Markdown:

**1️⃣ Confidence Assessment**
Start with: {confidence_status}
{"⚠️ **ALERT**: Low confidence detected - Possible causes: Complex lighting, non-standard paper color, intricate folding patterns. Review top-3 alternatives." if confidence < 25 else ""}
{"🟡 **NOTE**: Moderate confidence - Consider the alternative predictions below." if 25 <= confidence < 60 else ""}

**2️⃣ Top-3 Predictions & Reasoning**
List the 3 models with one-sentence geometric reasoning each (similarity in fold patterns, structural features, etc)

**3️⃣ Geometric Analysis**
Provide expert technical commentary on:
- Fold line structure and symmetry patterns
- Central folding mechanics and reference points
- Edge characteristics and finishing style
- Overall geometric composition and balance

**4️⃣ Technical Specifications**
Create a clean Markdown table with:
| Specification | Value |
| --- | --- |
| Model | {model_name} |
| Creator | {creator} |
| Difficulty | {formatted_difficulty} |
| Paper Shape | {paper_shape} |
| Cutting Required | {"Yes" if uses_cutting else "No"} |
| Glue Required | {"Yes" if uses_glue else "No"} |

**5️⃣ Expert Recommendation**
One motivating sentence to inspire the user. If confidence is low, mention trying this model as a learning opportunity.

TONE: Professional, technical, encouraging. Use origami terminology precisely."""

    return client.generate_response(prompt, temperature=0.7, max_tokens=650)


if __name__ == "__main__":
    # Test Groq connection
    client = get_groq_client()
    if client and client.is_available():
        print("✅ Groq API is ready!")
        test_response = client.generate_response(
            "What is origami?", max_tokens=100
        )
        if test_response:
            print(f"\n📝 Sample Response:\n{test_response}")
    else:
        print("❌ Groq API is not configured properly")
