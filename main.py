from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
import re

# Load environment variables
load_dotenv()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-flash-latest")

app = FastAPI(
    title="Pet Health AI Assistant",
    description="Analyze pet images and data with Google Gemini AI.",
    version="1.0.0"
)

# --- Models ---
class PetData(BaseModel):
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    grooming_frequency_per_week: Optional[int] = None
    meal_frequency_per_day: Optional[int] = None
    walk_frequency_per_day: Optional[int] = None
    age_in_years: int
    age_in_months: int


class NormalValues(BaseModel):
    weight_kg: str
    height_cm: str
    grooming_frequency_per_week: str
    meal_frequency_per_day: str
    walk_frequency_per_day: str


class PetAnalysisResponse(BaseModel):
    breed: str
    user_input_summary: Dict[str, Union[float, int, str, None]]
    normal_values: NormalValues
    errors: Dict[str, str]
    suggestions: List[str]


# --- Utils ---
def extract_json_from_text(text: str) -> str:
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})", text)
    if match:
        return match.group(1) or match.group(2)
    raise ValueError("No valid JSON object found in Gemini response.")


def parse_range_or_value(normal_str: str):
    match = re.match(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", normal_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.match(r"(\d+(?:\.\d+)?)", normal_str)
    if match:
        val = float(match.group(1))
        return val, val
    return None, None


def validate_user_values(user_values: Dict, normal_values: Dict) -> Dict[str, str]:
    results = {}
    for field, normal_str in normal_values.items():
        user_val = user_values.get(field)
        min_val, max_val = parse_range_or_value(normal_str)
        if user_val is not None and min_val is not None and max_val is not None:
            if min_val <= float(user_val) <= max_val:
                results[field] = "ok"
            else:
                results[field] = normal_str
        else:
            results[field] = normal_str  # Show normal values if not provided
    return results


def generate_suggestions(errors: Dict[str, str], user_values: Dict, breed: str) -> List[str]:
    suggestions = []
    for field, status in errors.items():
        user_val = user_values.get(field)
        if status != "ok":
            min_val, max_val = parse_range_or_value(status)
            field_name_friendly = field.replace('_', ' ').replace('per day', '/day').replace('per week', '/week')

            if user_val is None:
                suggestions.append(f"It looks like you haven't provided information for your pet's {field_name_friendly}. For a {breed}, the normal range is typically {status}.")
            elif min_val is not None and max_val is not None:
                if field == "weight_kg":
                    if user_val > max_val:
                        suggestions.append(
                            f"Your {breed} is currently weighing in at {user_val}kg, which is a bit above the normal range of {status}. Consider discussing a healthy weight management plan with your vet!"
                        )
                    elif user_val < min_val:
                        suggestions.append(
                            f"Your {breed} is currently weighing {user_val}kg, which is a little below the normal range of {status}. It might be good to chat with your vet about their diet to ensure they're getting enough nutrients."
                        )
                elif field == "meal_frequency_per_day":
                    if user_val < min_val:
                        suggestions.append(f"For a {breed}, a healthy meal frequency is usually {status}. You might consider increasing meal frequency slightly after consulting with your vet.")
                    elif user_val > max_val:
                        suggestions.append(f"For a {breed}, a healthy meal frequency is usually {status}. You might consider decreasing meal frequency slightly after consulting with your vet.")
                elif field == "walk_frequency_per_day":
                    if user_val < min_val:
                        suggestions.append(f"To keep your {breed} happy and healthy, aim for {status} walks/day. Regular exercise is key!")
                    elif user_val > max_val:
                        suggestions.append(f"Your {breed} is getting {user_val} walks/day! While exercise is great, the typical recommendation is {status}. Make sure your pet isn't overexerting themselves.")
                elif field == "grooming_frequency_per_week":
                    if user_val < min_val:
                        suggestions.append(f"For a {breed} like yours, we'd typically recommend grooming {status} times/week to keep their coat healthy and prevent matting.")
                    elif user_val > max_val:
                        suggestions.append(f"You're grooming your {breed} {user_val} times/week! The normal recommendation is {status}. While frequent grooming is good, ensure it's not irritating their skin.")
                elif field == "height_cm":
                    if user_val < min_val:
                        suggestions.append(f"Your {breed}'s height of {user_val}cm is a bit below the typical range of {status}. This could be normal for your individual pet, but it's always good to mention it to your vet.")
                    elif user_val > max_val:
                        suggestions.append(f"Your {breed}'s height of {user_val}cm is a bit above the typical range of {status}. This could be normal for your individual pet, but it's always good to mention it to your vet.")

    suggestions.extend([
        f"To ensure your {breed} thrives, focus on a balanced diet and consistent exercise tailored to their needs. They'll thank you for it!",
        "Don't forget the importance of regular veterinary check-ups! They're crucial for catching any potential health issues early and keeping your beloved pet in tip-top shape."
    ])
    return suggestions


def get_life_stage(age_years: int, age_months: int) -> str:
    total_months = age_years * 12 + age_months
    if total_months < 12:
        return "puppy/kitten"
    elif total_months < 96:
        return "adult"
    else:
        return "senior"


# --- Endpoints ---
@app.post("/analyze-pet", response_model=PetAnalysisResponse)
async def analyze_pet(
    pet_image: Optional[UploadFile] = File(None),
    breed_name: Optional[str] = Form(None),
    age_in_years: int = Form(...),
    age_in_months: int = Form(...),
    weight_kg: Optional[float] = Form(None),
    height_cm: Optional[float] = Form(None),
    grooming_frequency_per_week: Optional[int] = Form(None),
    meal_frequency_per_day: Optional[int] = Form(None),
    walk_frequency_per_day: Optional[int] = Form(None),
):
    if not pet_image and not breed_name:
        raise HTTPException(status_code=400, detail="Provide either pet image or breed name.")

    pet_data_obj = PetData(
        weight_kg=weight_kg,
        height_cm=height_cm,
        grooming_frequency_per_week=grooming_frequency_per_week,
        meal_frequency_per_day=meal_frequency_per_day,
        walk_frequency_per_day=walk_frequency_per_day,
        age_in_years=age_in_years,
        age_in_months=age_in_months
    )
    user_input_summary_data = pet_data_obj.model_dump(exclude_none=True)
    pet_data_str = json.dumps(user_input_summary_data, indent=2)

    life_stage = get_life_stage(age_in_years, age_in_months)

    # Initialize prompt with placeholders for breed and normal ranges
    prompt_template = """
You are a veterinary health assistant AI.
Return ONLY JSON with no extra text.

Breed: {breed_placeholder}
Age: {age_in_years} years, {age_in_months} months
Life Stage: {life_stage}

User Data:
{pet_data_str}

JSON Structure:
{{
  "breed": "<breed>",
  "normal_values": {{
    "weight_kg": "...",
    "height_cm": "...",
    "grooming_frequency_per_week": "...",
    "meal_frequency_per_day": "...",
    "walk_frequency_per_day": "..."
  }}
}}
"""

    contents = []
    actual_breed_for_prompt = "unknown" # Default
    
    if breed_name and breed_name.strip(): # Check if breed_name is provided and not just empty string
        actual_breed_for_prompt = breed_name.strip()
        # If breed name is provided, use it and don't rely on image for breed
        contents.append(prompt_template.format(
            breed_placeholder=actual_breed_for_prompt,
            age_in_years=age_in_years,
            age_in_months=age_in_months,
            life_stage=life_stage,
            pet_data_str=pet_data_str
        ))
        if pet_image: # If image is also provided, tell the model to ignore it for breed
            contents.append("The owner has explicitly stated the breed. Ignore the image for breed identification and focus on providing data for the stated breed.")
    elif pet_image:
        if pet_image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail="Invalid image type.")
        image_bytes = await pet_image.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Image too large.")
        
        # Insert image first for multimodal input
        contents.append({"mime_type": pet_image.content_type, "data": image_bytes})
        
        # Now add the prompt for breed identification from image
        contents.append(prompt_template.format(
            breed_placeholder="<identify_from_image>", # Placeholder to indicate breed should be identified
            age_in_years=age_in_years,
            age_in_months=age_in_months,
            life_stage=life_stage,
            pet_data_str=pet_data_str
        ))
        contents.append("Identify the breed from the image and then provide the correct normal ranges and suggestions for that breed.")
    else:
        # Fallback if neither image nor breed name is provided (though already handled by HTTPException)
        raise HTTPException(status_code=400, detail="Provide either pet image or breed name.")

    try:
        response = await model.generate_content_async(contents)
        json_string = extract_json_from_text(response.text.strip())
        response_data = json.loads(json_string)

        # Ensure the breed in the response is not "<identify_from_image>" if it was a placeholder
        if response_data.get("breed") == "<identify_from_image>" and pet_image:
            # If the model didn't infer the breed, and we asked it to, it's an issue.
            # For now, let's assume the model will fill it. If it doesn't, this part
            # would need more robust error handling or default to "unknown"
            response_data["breed"] = "Unknown Pet" # Or re-attempt breed identification

        response_data["user_input_summary"] = user_input_summary_data
        normal_values_dict = response_data.get("normal_values", {})
        errors_dict = validate_user_values(user_input_summary_data, normal_values_dict)
        response_data["errors"] = errors_dict
        response_data["suggestions"] = generate_suggestions(errors_dict, user_input_summary_data, response_data.get("breed", "your pet"))

        return PetAnalysisResponse(**response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")