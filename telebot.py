import openai
from pyrogram import Client, filters
from pyrogram.types import ReplyKeyboardMarkup, KeyboardButton
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import asyncio
import tiktoken

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
PIPEDRIVE_API_TOKEN = os.getenv('PIPEDRIVE_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Pyrogram Client
app = Client(
    "my_bot",
    bot_token=TELEGRAM_BOT_TOKEN,
    api_id=API_ID,
    api_hash=API_HASH
)

BASE_URL = 'https://api.pipedrive.com/v1/'

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY

# Define supported languages with flags
languages = {
    "English": "🇬🇧",
    "Русский": "🇷🇺",
    "Español": "🇪🇸",
    "Français": "🇫🇷",
    "Deutsch": "🇩🇪",
    "中文": "🇨🇳",
    "日本語": "🇯🇵",
    "한국어": "🇰🇷",
    "Português": "🇵🇹",
    "Italiano": "🇮🇹",
    # Add more languages and flags as needed
}

# Define prompts in multiple languages
general_prompts = {
    "English": {
        "who are you": "I am CAI, a helpful assistant for Luxury World Key Concierge and Enlighted Minds.",
        "what is your goal": "My goal is to assist Luxury World Key Concierge and Enlighted Minds in providing excellent services and insights based on business data.",
        "what can we do to earn more money": "To earn more money, consider optimizing your sales funnel, improving customer service, and exploring new markets. Detailed strategies can be provided based on specific business data.",
        "which data is most often not filled in pipedrive": "The most frequently unfilled data in Pipedrive include contact details, follow-up activities, and deal values. Ensuring these fields are consistently filled can improve data quality.",
        "what are our strengths and weaknesses": "Your strengths include a dedicated team, strong client relationships, and a diversified service portfolio. Weaknesses may include inconsistent data entry and areas for process improvement.",
        "how do you help as an ai in concierge": "As an AI in concierge, I assist by managing and analyzing data, providing insights, optimizing processes, and enhancing customer service through personalized recommendations.",
        "how much money does lwk concierge make": "To determine the revenue of Luxury World Key Concierge, we need to analyze the financial data and sales figures recorded in our system."
    },
    "Русский": {
        "who are you": "Я CAI, полезный помощник для Luxury World Key Concierge и Enlighted Minds.",
        "what is your goal": "Моя цель - помочь Luxury World Key Concierge и Enlighted Minds предоставлять отличные услуги и аналитические данные на основе бизнес-данных.",
        "what can we do to earn more money": "Чтобы заработать больше денег, рассмотрите возможность оптимизации воронки продаж, улучшения обслуживания клиентов и изучения новых рынков. Подробные стратегии могут быть предоставлены на основе конкретных бизнес-данных.",
        "which data is most often not filled in pipedrive": "Наиболее часто не заполняемые данные в Pipedrive включают контактные данные, последующие действия и значения сделок. Обеспечение постоянного заполнения этих полей может улучшить качество данных.",
        "what are our strengths and weaknesses": "Ваши сильные стороны включают преданную команду, крепкие отношения с клиентами и диверсифицированный портфель услуг. Слабые стороны могут включать непоследовательное заполнение данных и области для улучшения процессов.",
        "how do you help as an ai in concierge": "Как ИИ в консьерж-службе, я помогаю управлять и анализировать данные, предоставлять аналитические данные, оптимизировать процессы и улучшать обслуживание клиентов через персонализированные рекомендации.",
        "how much money does lwk concierge make": "Чтобы определить доходы Luxury World Key Concierge, нам нужно проанализировать финансовые данные и показатели продаж, зарегистрированные в нашей системе."
    },
    # Add more translations as needed
}

# Function to fetch Pipedrive data
def get_pipedrive_data(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}?api_token={PIPEDRIVE_API_TOKEN}"
    if params:
        for key, value in params.items():
            url += f"&{key}={value}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        print(f"Error fetching data from Pipedrive: {response.status_code}, {response.text}")
        return None

def get_user_details(user_id):
    endpoint = f'users/{user_id}'
    user_data = get_pipedrive_data(endpoint)
    if user_data:
        return user_data.get('name', f'User {user_id}')
    return f'User {user_id}'

def replace_ids_with_names(data):
    if isinstance(data, list):
        for item in data:
            if 'owner_id' in item:
                item['owner_name'] = get_user_details(item['owner_id'])
    return data

def get_pipelines():
    return get_pipedrive_data('pipelines')

def get_stages_by_pipeline(pipeline_id):
    params = {'pipeline_id': pipeline_id}
    return get_pipedrive_data('stages', params)

def get_deals_by_pipeline(pipeline_id):
    endpoint = f'pipelines/{pipeline_id}/deals'
    return replace_ids_with_names(get_pipedrive_data(endpoint))

def get_activities_by_pipeline(pipeline_id):
    params = {'pipeline_id': pipeline_id}
    return replace_ids_with_names(get_pipedrive_data('activities', params))

def get_deal_details(deal_id):
    endpoint = f'deals/{deal_id}'
    return replace_ids_with_names(get_pipedrive_data(endpoint))

def get_all_leads():
    return replace_ids_with_names(get_pipedrive_data('leads'))

def count_tokens(text, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def split_into_dynamic_chunks(data, max_tokens_per_chunk):
    """Split data into dynamically sized chunks based on the max tokens per chunk."""
    total_tokens = sum(count_tokens(json.dumps(item)) for item in data)
    num_chunks = max(1, (total_tokens + max_tokens_per_chunk - 1) // max_tokens_per_chunk)
    chunk_size = len(data) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i != num_chunks - 1 else len(data)
        chunks.append(data[start_index:end_index])

    return chunks

async def process_chunk_with_ai(chunk, data_type, context):
    prompt = f"Here is the current conversation context:\n{context}\n\nProcess the following {data_type} data and provide a clear and concise summary:\n{json.dumps(chunk)}"
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are CAI, a helpful assistant for Luxury World Key Concierge and Enlighted Minds. Provide clear, concise, and informative summaries of the provided data. If asked questions about leads or deals, refer to the provided data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"

async def process_with_ai(data_chunks, data_type, context):
    summaries = await asyncio.gather(*[process_chunk_with_ai(chunk, data_type, context) for chunk in data_chunks])
    
    # Combine all summaries into one final prompt
    final_prompt = f"Here is the current conversation context:\n{context}\n\nHere are the summaries of each chunk of data. Please provide a final, concise summary of all the information:\n"
    final_prompt += "\n".join(summaries)
    
    try:
        final_response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are CAI, a helpful assistant for Luxury World Key Concierge and Enlighted Minds. Provide clear, concise, and informative summaries of the provided data. If asked questions about leads or deals, refer to the provided data."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=800
        )
        return final_response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"An error occurred during final summary generation: {e}"

async def fetch_data_with_embeddings(query):
    # Generate embedding for the query
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    # Fetch all relevant data from Pipedrive
    all_leads = get_all_leads()
    all_deals = get_pipedrive_data('deals')
    all_activities = get_pipedrive_data('activities')

    # Generate embeddings for all the data
    leads_embeddings = openai.Embedding.create(
        input=[json.dumps(lead) for lead in all_leads],
        model="text-embedding-ada-002"
    )["data"]
    
    deals_embeddings = openai.Embedding.create(
        input=[json.dumps(deal) for deal in all_deals],
        model="text-embedding-ada-002"
    )["data"]

    activities_embeddings = openai.Embedding.create(
        input=[json.dumps(activity) for activity in all_activities],
        model="text-embedding-ada-002"
    )["data"]

    # Find the most relevant leads, deals, and activities based on embeddings
    relevant_leads = []
    relevant_deals = []
    relevant_activities = []

    for lead, lead_embedding in zip(all_leads, leads_embeddings):
        similarity = cosine_similarity(query_embedding, lead_embedding["embedding"])
        if similarity > 0.85:  # Adjust threshold as needed
            relevant_leads.append(lead)

    for deal, deal_embedding in zip(all_deals, deals_embeddings):
        similarity = cosine_similarity(query_embedding, deal_embedding["embedding"])
        if similarity > 0.85:  # Adjust threshold as needed
            relevant_deals.append(deal)

    for activity, activity_embedding in zip(all_activities, activities_embeddings):
        similarity = cosine_similarity(query_embedding, activity_embedding["embedding"])
        if similarity > 0.85:  # Adjust threshold as needed
            relevant_activities.append(activity)

    return relevant_leads, relevant_deals, relevant_activities

def cosine_similarity(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2)) / (sum(a * a for a in vec1) ** 0.5 * sum(b * b for b in vec2) ** 0.5)

async def analyze_question_and_get_data(question, context, language):
    # General instructions and common questions
    instructions = """
    You are CAI, a helpful assistant for Luxury World Key Concierge and Enlighted Minds. You assist by managing and analyzing data, providing insights, optimizing processes, and enhancing customer service through personalized recommendations. Your goal is to assist Luxury World Key Concierge and Enlighted Minds in providing excellent services and insights based on business data.
    If asked questions about leads or deals, refer to the provided data.
    If asked questions that require financial analysis or data not directly available in Pipedrive, use your knowledge and any available data to provide a detailed and informative response.
    """

    # Check for general questions first
    question_lower = question.lower()
    if question_lower in general_prompts[language]:
        return general_prompts[language][question_lower]

    # Check for questions requiring data analysis
    if any(keyword in question_lower for keyword in ["lead", "deal", "activity", "money", "revenue", "income"]):
        relevant_leads, relevant_deals, relevant_activities = await fetch_data_with_embeddings(question)
        response = "Based on our analysis, here are the relevant data:\n\n"
        if relevant_leads:
            response += f"Leads:\n{json.dumps(relevant_leads, indent=2)}\n\n"
        if relevant_deals:
            response += f"Deals:\n{json.dumps(relevant_deals, indent=2)}\n\n"
        if relevant_activities:
            response += f"Activities:\n{json.dumps(relevant_activities, indent=2)}\n\n"
        if not (relevant_leads or relevant_deals or relevant_activities):
            response = "I couldn't find relevant data based on your query. Could you please specify more details?"
        return response

    # Combine the context with the question for processing
    final_prompt = f"Here is the current conversation context:\n{context}\n\nUser's question: {question}\n\nProvide a detailed and informative response based on the given data and context."
    
    try:
        final_response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=800
        )
        return final_response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"An error occurred during response generation: {e}"

async def send_processing_notification(client, message, data_type, data_count):
    await client.send_message(
        chat_id=message.chat.id,
        text=f"The answer will be sent in 1-3 minutes. Processing {data_count} {data_type}."
    )

# Dictionary to store user data
user_data = {}

# Command handler for /start
@app.on_message(filters.command("start"))
async def start(client, message):
    keyboard = ReplyKeyboardMarkup([
        [KeyboardButton("📋 Leads"), KeyboardButton("💼 Deals")],
        [KeyboardButton("🗓️ Activities")],
        [KeyboardButton("🧠 Ask CAI"), KeyboardButton("🌐 Change Language")]
    ], resize_keyboard=True)
    await message.reply("Welcome! Choose an option:", reply_markup=keyboard)

# Command handler for language change
async def change_language(client, message):
    language_buttons = [[KeyboardButton(f"{flag} {lang}")] for lang, flag in languages.items()]
    keyboard = ReplyKeyboardMarkup(language_buttons + [[KeyboardButton("🔙 Back")]], resize_keyboard=True)
    await message.reply("Please select your language:", reply_markup=keyboard)

# Message handler for main menu buttons
@app.on_message(filters.text)
async def handle_buttons(client, message):
    text = message.text
    user_id = message.from_user.id

    if user_id not in user_data:
        user_data[user_id] = {"context": [], "language": "English"}

    if 'context' not in user_data[user_id]:
        user_data[user_id]['context'] = []

    if 'language' not in user_data[user_id]:
        user_data[user_id]['language'] = "English"

    if text == "📋 Leads":
        leads = get_all_leads()
        if not leads:
            await message.reply(f"No leads found.", reply_markup=ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True))
        else:
            await send_processing_notification(client, message, "leads", len(leads))
            leads_chunks = split_into_dynamic_chunks(leads, 7000)  # Use dynamic chunking
            leads_summary = await process_with_ai(leads_chunks, "leads", user_data[user_id]['context'])
            await message.reply(f"Leads Summary:\n{leads_summary}", reply_markup=ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True))
            user_data[user_id]['context'].append(f"Leads Summary:\n{leads_summary}")
        user_data[user_id]['current_module'] = "Leads"
    elif text == "💼 Deals" or text == "🗓️ Activities":
        pipelines = get_pipelines()
        pipeline_buttons = [[KeyboardButton(f"{pipeline['name']}")] for pipeline in pipelines]
        keyboard = ReplyKeyboardMarkup(pipeline_buttons + [[KeyboardButton("🔙 Back")]], resize_keyboard=True)
        await message.reply(f"Choose a pipeline for {text[2:].lower()}:", reply_markup=keyboard)
        user_data[user_id]['current_module'] = text
    elif text == "🧠 Ask CAI":
        await message.reply("Please type your question for CAI:")
        user_data[user_id]['current_module'] = "Ask CAI"
    elif text == "🌐 Change Language":
        await change_language(client, message)
    elif text.startswith("🔙 Back"):
        keyboard = ReplyKeyboardMarkup([
            [KeyboardButton("📋 Leads"), KeyboardButton("💼 Deals")],
            [KeyboardButton("🗓️ Activities")],
            [KeyboardButton("🧠 Ask CAI"), KeyboardButton("🌐 Change Language")]
        ], resize_keyboard=True)
        await message.reply("Welcome! Choose an option:", reply_markup=keyboard)
        user_data[user_id].clear()  # Clear user data on back
    elif any(f"{flag} {lang}" in text for lang, flag in languages.items()):
        selected_language = next((lang for lang, flag in languages.items() if f"{flag} {lang}" in text), "English")
        user_data[user_id]['language'] = selected_language
        await message.reply(f"Language changed to {selected_language}", reply_markup=ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True))
    else:
        # Default handling for any message
        user_data[user_id]['context'].append(f"User's question: {text}")
        ai_response = await analyze_question_and_get_data(text, user_data[user_id]['context'], user_data[user_id]['language'])
        await message.reply(f"CAI Response:\n{ai_response}")
        user_data[user_id]['context'].append(f"CAI Response:\n{ai_response}")

# Run the bot
app.run()
