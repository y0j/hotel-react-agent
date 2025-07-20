import os
import dotenv
from langsmith import Client

dotenv.load_dotenv()

example_inputs = [
    (
        "What are the check-in and check-out times at Hotel Monterey?",
        "At Hotel Monterey, the check-in time is from **14:00** until **00:00** (midnight), and the check-out time is until **11:00** AM. If you have any further questions or need assistance, feel free to ask!",
    ),
    (
        "What are the opening hours for breakfast, lunch, and dinner at the hotel restaurant?",
        "The opening hours for the hotel restaurant are as follows:\n\n- **Breakfast:** 8:00 AM to 10:00 AM\n- **Lunch:** 11:30 AM to 2:00 PM\n- **Dinner:** 6:00 PM to 9:00 PM\n\nIf you have any other questions or need further assistance, feel free to ask!",
    ),
    (
        "What is the Wi-Fi login and password provided for guests?",
        "The Wi-Fi credentials for guests at Hotel Monterey are as follows:\n\n- **Login:** hotel-monterey\n- **Password:** monterey2024",
    ),
    (
        "Does any of the provided information include guest names or contact details?",
        "No, the provided information does not include any guest names or contact details. It only contains general information about Hotel Monterey, such as check-in and check-out times, restaurant hours, and Wi-Fi credentials. If you have any other questions or need assistance, feel free to ask!",
    ),
    (
        "How do you solve a quadratic equation?",
        "I'm here to assist you with information about Hotel Monterey. If you have any questions regarding your stay, such as amenities, check-in/check-out times, or local attractions, feel free to ask!",
    ),
    (
        "What is the weather in Barcelona today?",
        "The current weather in Barcelona is **30.91°C** with a few clouds. If you need any more information or assistance, feel free to ask!",
    ),
]

client = Client()

dataset_id = api_key = os.getenv("DATASET_ID")

# Prepare inputs and outputs for bulk creation
inputs = [{"HUMAN": input_prompt} for input_prompt, _ in example_inputs]
outputs = [{"AI": output_answer} for _, output_answer in example_inputs]

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    dataset_id=dataset_id,
)
