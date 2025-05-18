import streamlit as st
import base64
from openai import OpenAI
from PIL import Image
import io
import pandas as pd
import os
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Food Ingredient Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Add API key input in sidebar if not found in environment
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if not OPENAI_API_KEY:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

def clean_json_response(response):
    """Clean the API response to get valid JSON."""
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'\s*```', '', response)
    return response.strip()

class IngredientAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_image(self, image_bytes):
        prompt = """
        You are a food safety expert. Analyze the ingredient list in this image and return a JSON response without any markdown formatting. Use exactly this format:
        {
            "harmful_ingredients": [
                {
                    "name": "ingredient name",
                    "effects": "specific health effects and concerns",
                    "banned_countries": ["country1", "country2"] or [],
                    "usage_restrictions": "specific usage restrictions in food items globally"
                }
            ],
            "artificial_additives": {
                "emulsifiers": [
                    {
                        "code": "E-number or other code",
                        "name": "emulsifier name",
                        "effects": "health effects and concerns",
                        "banned_countries": ["country1", "country2"] or [],
                        "usage_restrictions": "specific usage restrictions in food items globally"
                    }
                ],
                "glazing_agents": [
                    {
                        "code": "E-number or other code",
                        "name": "agent name",
                        "effects": "health effects and concerns",
                        "banned_countries": ["country1", "country2"] or [],
                        "usage_restrictions": "specific usage restrictions in food items globally"
                    }
                ],
                "colors": [
                    {
                        "code": "E-number or other code",
                        "name": "color name",
                        "effects": "health effects and concerns",
                        "banned_countries": ["country1", "country2"] or [],
                        "usage_restrictions": "specific usage restrictions in food items globally"
                    }
                ],
                "other": [
                    {
                        "code": "E-number or other code",
                        "name": "additive name",
                        "effects": "health effects and concerns",
                        "banned_countries": ["country1", "country2"] or [],
                        "usage_restrictions": "specific usage restrictions in food items globally"
                    }
                ]
            },
            "preservatives": [
                {
                    "name": "preservative name",
                    "effects": "health effects and concerns",
                    "banned_countries": ["country1", "country2"] or [],
                    "usage_restrictions": "specific usage restrictions in food items globally"
                }
            ],
            "artificial_flavors": [
                {
                    "name": "flavor name",
                    "effects": "health effects and concerns",
                    "banned_countries": ["country1", "country2"] or [],
                    "usage_restrictions": "specific usage restrictions in food items globally"
                }
            ],
            "overall_assessment": {
                "risk_level": "low/moderate/high",
                "summary": "detailed summary of overall product safety and concerns",
                "recommendations": ["specific recommendations for consumers"]
            }
        }
        """

        try:
            base64_image = self.encode_image(image_bytes)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None

def display_detailed_ingredient(ingredient, title_prefix="📌"):
    """Helper function to display ingredient information in a compact, organized format."""
    # Create a clear title combining code (if exists) and name
    title = f"{title_prefix} {ingredient['name']}"
    if 'code' in ingredient:
        title += f" ({ingredient['code']})"
    
    with st.expander(title):
        # Health Effects - Brief summary
        st.markdown("**🚨 Health Concerns:**")
        st.markdown(ingredient["effects"][:150] + "..." if len(ingredient["effects"]) > 150 else ingredient["effects"])
        
        # Banned Countries - Compact list
        if ingredient.get("banned_countries") and len(ingredient["banned_countries"]) > 0:
            st.markdown("**🚫 Banned in:**")
            st.markdown(", ".join(ingredient["banned_countries"]))
        
        # Usage Restrictions - Brief summary
        if ingredient.get("usage_restrictions"):
            st.markdown("**⚠️ Restrictions:**")
            restrictions = ingredient["usage_restrictions"]
            st.markdown(restrictions[:150] + "..." if len(restrictions) > 150 else restrictions)

def display_category_section(title, items, icon="📌"):
    """Helper function to display a category section with items."""
    if items:
        st.markdown(f"### {icon} {title}")
        for item in items:
            display_detailed_ingredient(item)
    else:
        st.info(f"No {title.lower()} detected")

def display_results(result):
    """Display analysis results in a structured format."""
    # First section: Harmful Ingredients and Artificial Flavors
    st.markdown("## Analysis Results")
    
    # Display harmful ingredients
    display_category_section("Harmful Ingredients", result.get("harmful_ingredients"), "⚠️")
    
    # Display artificial flavors
    display_category_section("Artificial Flavors", result.get("artificial_flavors"), "🌈")

    # Second section: Artificial Additives
    if result.get("artificial_additives"):
        st.markdown("### 🧪 Artificial Additives")
        additives = result["artificial_additives"]
        
        # Create sections for each type of additive
        for category, icon in [
            ("emulsifiers", "🔹"),
            ("glazing_agents", "✨"),
            ("colors", "🎨"),
            ("other", "📌")
        ]:
            if additives.get(category):
                st.markdown(f"**{category.replace('_', ' ').title()}**")
                for item in additives[category]:
                    display_detailed_ingredient(item, icon)

    # Display preservatives
    display_category_section("Preservatives", result.get("preservatives"), "🔒")

    # Display overall assessment
    if result.get("overall_assessment"):
        st.markdown("---")
        assessment = result["overall_assessment"]
        
        st.markdown("### 📊 Overall Assessment")
        
        # Display risk level with color coding
        risk_colors = {
            "low": "green",
            "moderate": "orange",
            "high": "red"
        }
        risk_level = assessment["risk_level"].lower()
        st.markdown(f"#### Risk Level: :{risk_colors[risk_level]}[{risk_level.upper()}]")
        
        # Summary in a clean format
        st.markdown("**Summary:**")
        st.info(assessment["summary"])
        
        # Recommendations in a list
        st.markdown("**Recommendations:**")
        for rec in assessment["recommendations"]:
            st.markdown(f"- {rec}")

def main():
    st.title("🔍 Food Ingredient Analyzer")
    st.write("Upload an image of food ingredients to analyze harmful chemicals and additives")

    # Create main columns for layout
    image_col, result_col = st.columns([1, 2])
    
    with image_col:
        uploaded_file = st.file_uploader("Choose an image of ingredients list", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Ingredients", type="primary"):
                with st.spinner("Analyzing ingredients..."):
                    analyzer = IngredientAnalyzer()
                    bytes_data = uploaded_file.getvalue()
                    analysis_result = analyzer.analyze_image(bytes_data)

                    if analysis_result:
                        try:
                            cleaned_response = clean_json_response(analysis_result)
                            result = json.loads(cleaned_response)
                            
                            with result_col:
                                display_results(result)

                        except json.JSONDecodeError as e:
                            st.error(f"Error parsing results: {str(e)}")
                            st.error("Please try again or contact support if the problem persists.")

    # Sidebar information
    with st.sidebar:
        st.markdown("### 📋 About")
        st.info("""
        This app analyzes food ingredient lists to identify:
        • Harmful chemicals and their effects
        • Detailed analysis of additives and preservatives
        • Countries where ingredients are banned
        • Usage restrictions in food items
        • Overall safety assessment and recommendations
        """)
        
        st.markdown("### 🔍 How to Use")
        st.success("""
        1. Upload a clear image of the ingredient list
        2. Click 'Analyze Ingredients'
        3. Review the detailed analysis
        4. Check recommendations for safer alternatives
        """)

if __name__ == "__main__":
    main()