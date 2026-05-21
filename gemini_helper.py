from google import genai

# paste your API key here
client = genai.Client(api_key="AIzaSyDpWf-xdWNtkhTOAEXIAKWh0Co6IsVjFXA")

def analyze_resume_with_gemini(resume_text, job_desc, missing_skills):

    prompt = f"""
Resume:
{resume_text}

Job Description:
{job_desc}

Missing Skills:
{missing_skills}

Provide:
1. Resume weaknesses
2. Learning roadmap
3. Suggestions to improve resume
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text

    except Exception as e:
        return "Gemini AI temporarily unavailable due to quota limit."