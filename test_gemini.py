from google import genai

client = genai.Client(api_key="AIzaSyDpWf-xdWNtkhTOAEXIAKWh0Co6IsVjFXA")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain machine learning simply"
)

print(response.text)