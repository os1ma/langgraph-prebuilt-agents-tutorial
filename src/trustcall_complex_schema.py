from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from trustcall import create_extractor

load_dotenv(override=True)


class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str


class TelegramPreferences(BaseModel):
    preferred_encoding: list[OutputFormat] | None = None
    favorite_telegram_operators: list[OutputFormat] | None = None
    preferred_telegram_paper: list[OutputFormat] | None = None


class MorseCode(BaseModel):
    preferred_key_type: list[OutputFormat] | None = None
    favorite_morse_abbreviations: list[OutputFormat] | None = None


class Semaphore(BaseModel):
    preferred_flag_color: list[OutputFormat] | None = None
    semaphore_skill_level: list[OutputFormat] | None = None


class TrustFallPreferences(BaseModel):
    preferred_fall_height: list[OutputFormat] | None = None
    trust_level: list[OutputFormat] | None = None
    preferred_catching_technique: list[OutputFormat] | None = None


class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore


class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences


class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences


llm = ChatOpenAI(model="gpt-4o")

conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

bound = create_extractor(
    llm,
    tools=[TelegramAndTrustFallPreferences],
    tool_choice="TelegramAndTrustFallPreferences",
)

result = bound.invoke(
    f"""Extract the preferences from the following conversation:
<convo>
{conversation}
</convo>""",
)
print(result["responses"][0])
