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
    """Helper function to display detailed ingredient information."""
    with st.expander(f"{title_prefix} {ingredient['name']}"):
        if 'code' in ingredient:
            st.markdown(f"**Code:** {ingredient['code']}")
        
        st.markdown("#### ⚕️ Health Effects")
        st.write(ingredient["effects"])
        
        st.markdown("#### 🚫 Banned in Countries")
        if ingredient.get("banned_countries") and len(ingredient["banned_countries"]) > 0:
            st.write(", ".join(ingredient["banned_countries"]))
        else:
            st.write("No known bans reported")
        
        if ingredient.get("usage_restrictions"):
            st.markdown("#### ⚠️ Usage Restrictions")
            st.write(ingredient["usage_restrictions"])

def display_results(result):
    """Display analysis results in a structured format."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display harmful ingredients
        if result.get("harmful_ingredients"):
            st.markdown("### ⚠️ Harmful Ingredients Detected")
            for ingredient in result["harmful_ingredients"]:
                display_detailed_ingredient(ingredient)
        else:
            st.success("No harmful ingredients detected")

        # Display artificial flavors
        if result.get("artificial_flavors"):
            st.markdown("### 🌈 Artificial Flavors")
            for flavor in result["artificial_flavors"]:
                display_detailed_ingredient(flavor)
        else:
            st.info("No artificial flavors detected")

    with col2:
        # Display artificial additives
        if result.get("artificial_additives"):
            st.markdown("### 🧪 Artificial Additives")
            
            additives = result["artificial_additives"]
            
            # Display emulsifiers
            if additives.get("emulsifiers"):
                st.markdown("#### Emulsifiers")
                for emulsifier in additives["emulsifiers"]:
                    display_detailed_ingredient(emulsifier, "🔹")

            # Display glazing agents
            if additives.get("glazing_agents"):
                st.markdown("#### Glazing Agents")
                for agent in additives["glazing_agents"]:
                    display_detailed_ingredient(agent, "🔹")

            # Display colors
            if additives.get("colors"):
                st.markdown("#### Colors")
                for color in additives["colors"]:
                    display_detailed_ingredient(color, "🔹")

            # Display other additives
            if additives.get("other"):
                st.markdown("#### Other Additives")
                for additive in additives["other"]:
                    display_detailed_ingredient(additive, "🔹")
        else:
            st.info("No artificial additives detected")

        # Display preservatives
        if result.get("preservatives"):
            st.markdown("### 🔒 Preservatives")
            for preservative in result["preservatives"]:
                display_detailed_ingredient(preservative)
        else:
            st.info("No preservatives detected")

    # Display overall assessment
    if result.get("overall_assessment"):
        st.markdown("---")
        st.markdown("### 📊 Overall Assessment")
        
        assessment = result["overall_assessment"]
        risk_level = assessment["risk_level"].lower()
        
        # Display risk level with appropriate color
        if risk_level == "low":
            st.markdown("#### Risk Level: :green[LOW]")
        elif risk_level == "moderate":
            st.markdown("#### Risk Level: :orange[MODERATE]")
        else:
            st.markdown("#### Risk Level: :red[HIGH]")
        
        st.markdown("#### Summary")
        st.write(assessment["summary"])
        
        st.markdown("#### Recommendations")
        for rec in assessment["recommendations"]:
            st.markdown(f"- {rec}")

def main():
    st.title("🔍 Food Ingredient Analyzer")
    st.write("Upload an image of food ingredients to analyze harmful chemicals and additives")

    uploaded_file = st.file_uploader("Choose an image of ingredients list", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Ingredients"):
            with st.spinner("Analyzing ingredients..."):
                analyzer = IngredientAnalyzer()
                bytes_data = uploaded_file.getvalue()
                analysis_result = analyzer.analyze_image(bytes_data)

                if analysis_result:
                    try:
                        cleaned_response = clean_json_response(analysis_result)
                        result = json.loads(cleaned_response)
                        
                        with col2:
                            display_results(result)

                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing results: {str(e)}")
                        st.error("Please try again or contact support if the problem persists.")

    with st.sidebar:
        st.markdown("### About")
        st.write("""
        This app analyzes food ingredient lists to identify:
        - Harmful chemicals and their effects
        - Detailed analysis of additives and preservatives
        - Countries where ingredients are banned
        - Usage restrictions in food items
        - Overall safety assessment and recommendations
        """)
        
        st.markdown("### How to Use")
        st.write("""
        1. Upload a clear image of the ingredient list
        2. Click 'Analyze Ingredients'
        3. Review the detailed analysis
        4. Check recommendations for safer alternatives
        """)

if __name__ == "__main__":
    main()