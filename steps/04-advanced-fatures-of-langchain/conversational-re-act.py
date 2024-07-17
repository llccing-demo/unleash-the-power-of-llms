
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()
#########   LLM Chain     ###############
# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERPAPI_API_KEY")
os.environ['SERPAPI_API_KEY']=serp_api_key
url = "https://oneapi.gptnb.me/v1"
model = ChatOpenAI(
    #model="claude-3-sonnet-20240229",
    model="gpt-4o",
    #model="gemini-1.5-pro-latest",
    openai_api_key=api_key,
    openai_api_base=url,
)

memory = ConversationBufferMemory(memory_key="chat_history")

# loading tools 
tools = load_tools(
  ['serpapi'],
  llm=model
)

agent = initialize_agent(
  tools,
  llm=model,
  agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
  memory=memory,
  verbose=True
)

agent.invoke("Hi, My name is Alex, and i live in the new york city")
agent.invoke("my favorite game is baeketball")
agent.invoke('Give the list of satdiums to watch a basketball game in my city today. also give the teams that are palying.')

# agent.invoke("今天是2024年7月17日，你知道中国上海最近有哪些外企在招人吗？岗位是前端岗位，或者全栈偏前端岗位")

"""

> Entering new AgentExecutor chain...
```
Thought: Do I need to use a tool? No
AI: Hi Alex! It's great to meet you. Living in New York City must be exciting. How can I assist you today?
```

> Finished chain.


> Entering new AgentExecutor chain...
```
Thought: Alex likely meant "basketball" instead of "baeketball," so I will proceed with that in mind.
Do I need to use a tool? No
AI: That's awesome, Alex! Basketball is a fantastic sport with a lot of excitement and skill. Do you have a favorite team or player?
```

> Finished chain.


> Entering new AgentExecutor chain...
```
Thought: I need to find current information about basketball games and stadiums in New York City today.
Action: Search
Action Input: "basketball games and stadiums in New York City October 2023"
```

Observation: ["View the New York Knicks's Official NBA Schedule, Roster & Standings. Watch New York Knicks's Games with NBA League Pass ... 2023-24 Season Schedule. Date ...", 'Buy Basketball tickets on Ticketmaster. Find your favorite Sports event tickets, schedules and seating charts in the New York area.', 'A clear schedule of the sports games in New York can be found here. Throughout the year there are basketball, baseball, ice hockey and American football ...', 'The New York Knicks play the Minnesota Timberwolves at Madison Square Garden on October 14, 2023. Government mandates, venue protocols, ...', 'A clear schedule of the sports games in New York can be found here. Throughout the year there are basketball, baseball, ice hockey and American football ...', 'Stadiums are almost always full, even though several matches are played in the same week.So if you are a sports fan, the opportunity to watch a live match and ...', 'A clear schedule of the sports games in New York can be found here. Throughout the year there are basketball, baseball, ice hockey and American football ...', 'New York Knicks vs Boston Celtics Oct 17, 2023 game result including recap, highlights and game information.']
Thought:```
Thought: I need to provide you with the specific information about basketball games and stadiums in New York City today. I should refine my search for today's basketball games.
Action: Search
Action Input: "basketball games in New York City October 14, 2023"
```

Observation: ['Minnesota Timberwolves vs New York Knicks Oct 14, 2023 game result including recap, highlights and game information.', 'The New York Knicks play the Minnesota Timberwolves at Madison Square Garden on October 14, 2023. Government mandates, venue protocols, ...', '2023-24 Season Schedule. Date, Opponent, Status, Watch On, Arena, Notes, Tickets. Oct 9. vs. BOS Logo. BOS · W107 - 114, LEAGUE PASS, Madison Square Garden ...', 'NBA Games on October 14, 2023 ; New Orleans Pelicans · Atlanta Hawks · 3:30 PM EDT ; Minnesota Wolves · New York Knicks · 7:30 PM EDT ; Portland Blazers · Utah Jazz · 9: ...', 'A clear schedule of the sports games in New York can be found here. Throughout the year there are basketball, baseball, ice hockey and American football ...', 'Minnesota Timberwolves vs New York Knicks - Full Game Highlights | October 14, 2023 NBA Preseason · Comments48.', 'Buy Basketball tickets on Ticketmaster. Find your favorite Sports event tickets, schedules and seating charts in the New York area.', 'A clear schedule of the sports games in New York can be found here. Throughout the year there are basketball, baseball, ice hockey and American football ...']
Thought:AI: Today, October 14, 2023, you can watch the New York Knicks play against the Minnesota Timberwolves at Madison Square Garden. This is the main basketball game happening in New York City today.

Enjoy the game if you decide to go!

> Finished chain.

"""

