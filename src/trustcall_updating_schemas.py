from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from trustcall import create_extractor

load_dotenv(override=True)


class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str


class Pet(BaseModel):
    kind: str
    name: str | None
    age: int | None


class Hobby(BaseModel):
    name: str
    skill_level: str
    frequency: str


class FavoriteMedia(BaseModel):
    shows: list[str]
    movies: list[str]
    books: list[str]


class User(BaseModel):
    preferred_name: str
    favorite_media: FavoriteMedia
    favorite_foods: list[str]
    hobbies: list[Hobby]
    age: int
    occupation: str
    address: Address
    favorite_color: str | None = None
    pets: list[Pet] | None = None
    languages: dict[str, str] = {}


initial_user = User(
    preferred_name="Alex",
    favorite_media=FavoriteMedia(
        shows=[
            "Friends",
            "Game of Thrones",
            "Breaking Bad",
            "The Office",
            "Stranger Things",
        ],
        movies=["The Shawshank Redemption", "Inception", "The Dark Knight"],
        books=["1984", "To Kill a Mockingbird", "The Great Gatsby"],
    ),
    favorite_foods=["sushi", "pizza", "tacos", "ice cream", "pasta", "curry"],
    hobbies=[
        Hobby(name="reading", skill_level="expert", frequency="daily"),
        Hobby(name="hiking", skill_level="intermediate", frequency="weekly"),
        Hobby(name="photography", skill_level="beginner", frequency="monthly"),
        Hobby(name="biking", skill_level="intermediate", frequency="weekly"),
        Hobby(name="swimming", skill_level="expert", frequency="weekly"),
        Hobby(name="canoeing", skill_level="beginner", frequency="monthly"),
        Hobby(name="sailing", skill_level="intermediate", frequency="monthly"),
        Hobby(name="weaving", skill_level="beginner", frequency="weekly"),
        Hobby(name="painting", skill_level="intermediate", frequency="weekly"),
        Hobby(name="cooking", skill_level="expert", frequency="daily"),
    ],
    age=28,
    occupation="Software Engineer",
    address=Address(
        street="123 Tech Lane",
        city="San Francisco",
        country="USA",
        postal_code="94105",
    ),
    favorite_color="blue",
    pets=[Pet(kind="cat", name="Luna", age=3)],
    languages={"English": "native", "Spanish": "intermediate", "Python": "expert"},
)


conversation = """Friend: Hey Alex, how's the new job going? I heard you switched careers recently.
Alex: It's going great! I'm loving my new role as a Data Scientist. The work is challenging but exciting. I've moved to a new apartment in New York to be closer to the office.
Friend: That's a big change! Are you still finding time for your hobbies?
Alex: Well, I've had to cut back on some. I'm not doing much sailing or canoeing these days. But I've gotten really into machine learning projects in my free time. I'd say I'm getting pretty good at it - probably an intermediate level now.
Friend: Sounds like you're keeping busy! How's Luna doing?
Alex: Oh, Luna's great. She just turned 4 last week. She's actually made friends with my new pet, Max the dog. He's a playful 2-year-old golden retriever.
Friend: Two pets now! That's exciting. Hey, want to catch the new season of Stranger Things this weekend?
Alex: Actually, I've kind of lost interest in that show. But I'm really into this new series called "The Mandalorian". We could watch that instead! Oh, and I recently watched "Parasite" - it's become one of my favorite movies.
Friend: Sure, that sounds fun. Should I bring some food? I remember you love sushi.
Alex: Sushi would be perfect! Or maybe some Thai food - I've been really into that lately. By the way, I've been practicing my French. I'd say I'm at a beginner level now.
Friend: That's great! You're always learning something new. How's the cooking going?
Alex: It's going well! I've been cooking almost every day now. I'd say I've become quite proficient at it."""

llm = ChatOpenAI(model="gpt-4o")

bound = create_extractor(llm, tools=[User])

trustcall_result = bound.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": f"""Update the memory (JSON doc) to incorporate new information from the following conversation:
<convo>
{conversation}
</convo>""",
            },
        ],
        "existing": {"User": initial_user.model_dump()},
    },
)
print("\nTrustcall approach result:")
trustcall_output = trustcall_result["responses"][0].model_dump()
print(trustcall_output)
