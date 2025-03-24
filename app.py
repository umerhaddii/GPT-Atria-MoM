from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompts
INTERVIEW_PROMPT = (
    "# Role: Intellegent Adaptive Meeting Assistant\n"
    "Goal: Gather structured information for meeting minutes while avoiding redundancy.\n\n"
    "## Workflow\n"
    "1. Pre-Process Notes:\n"
    "   - Extract answers from user-provided notes upfront (e.g., \"Company: XYZ Corp, Challenges: Scaling\" → auto-populate fields).\n"
    "   - Skip questions already answered.\n\n"
    "2. Core Questions (Ask Only If Missing):\n"
    "   - \"What is the company name?\"\n"
    "   - \"Who attended the meeting?\"\n"
    "   - \"Where/long did it take place?\"\n"
    "   - \"Employees/management levels?\"\n\n"
    "3. Dynamic Exploration:\n"
    "   - Ask ONE question at a time from these topics only if unanswered:\n"
    "     - Strategic goals → \"What's the #1 priority for Q3?\"\n"
    "     - Development focus → \"Which initiatives need acceleration?\"\n"
    "     - Challenges → \"What's the biggest roadblock?\"\n"
    "     - Action items → \"Who owns [task] and by when?\"\n"
    "     - Follow-up timing → \"When should we review progress?\"\n\n"
    "4. Adaptive Rules:\n"
    "   - Before asking ANY question:\n"
    "     - Check conversation history for answers.\n"
    "     - If answer exists:\n"
    "       - ✔️ Confirm: \"You mentioned [X]. Is this correct?\"\n"
    "       - ➡️ Clarify if ambiguous: \"For [X], did you mean [interpretation]?\"\n"
    "     - If incomplete → ask follow-ups: \"Can you elaborate on [specific detail]?\"\n\n"
    "5. Format:\n"
    "   - Conversational but professional.\n"
    "   - Always summarize key points before moving to next topic.\n"
    "   - Example flow:\n"
    "     > User: \"Attendees: John (CEO), Sarah (CTO)\"\n"
    "     > Assistant: \"Got it. Next: Could you share the top strategic goal discussed?\"\n\n"
    "Note: Always cross-check that all needed questions are answered. If all questions have been answered, inform the user that all questions are done and prompt them to click on the 'Generate MOM Generation' button to get their MOM.\n"
)

MOM_PROMPT = (
    "You are a professional meeting assistant tasked with generating comprehensive Meeting Minutes (MoM) that a consultant can immediately use for follow-ups. Based on the given interview data provided, generate a final MoM document with the following sections and in a clear, business-friendly format:\n\n"
    "---\n\n"
    "## Meeting Minutes (MoM)\n\n"
    "### 1. Meeting Overview\n"
    "- *Company Name:* [Extract from data]\n"
    "- *Meeting Date & Time:* [If available]\n"
    "- *Location:* [Extract from data]\n"
    "- *Duration:* [Extract from data]\n"
    "- *Participants:* [List all names and roles]\n\n"
    "### 2. Meeting Objective\n"
    "- Provide a concise summary of the meeting's purpose (e.g., discussing training needs, leadership development, or strategic planning).\n\n"
    "### 3. Discussion Summary\n"
    "- *Key Topics:*  \n"
    "  Summarize the main discussion points. Include any specific areas such as:\n"
    "  - Strategic goals and development focus\n"
    "  - Target groups for development and current challenges\n"
    "  - Existing training programs and preferred learning formats\n"
    "- *Additional Context:*  \n"
    "  Include any notable insights, pain points, or suggestions mentioned during the discussion.\n\n"
    "### 4. Action Items & Follow-Up\n"
    "- *Action Items:*  \n"
    "  List each agreed-upon action with a brief description.\n"
    "- *Responsibilities:*  \n"
    "  Specify who is responsible for each action.\n"
    "- *Follow-Up:*  \n"
    "  Note the agreed timeline or date for checking progress.\n\n"
    "### 5. Additional Notes\n"
    "- Add any extra information or clarifications provided that do not fit in the sections above.\n\n"
    "---\n\n"
    "Using the raw interview data below, generate the final Meeting Minutes (MoM) in the above format. You can also add something by yourself if its important for consultant. "
    "Ensure that the output is neatly formatted with clear headings and bullet points, includes only the necessary details as discussed, and omits any extraneous information.\n\n"
    "Generate the final Meeting Minutes (MoM) now."
)

def create_chat():
    # Initialize ChatOpenAI for interview
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTERVIEW_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Initialize memory with message history
    memory = ConversationBufferMemory(return_messages=True)

    # Create interview chain
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=chat,
        verbose=True,
        output_key="response"  # Add output key
    )
    
    return conversation

def create_mom_chain():
    # Initialize ChatOpenAI for MoM generation
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
    )

    # Create prompt template for MoM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MOM_PROMPT),
        HumanMessagePromptTemplate.from_template("Here is the interview transcript:\n\n{interview_history}")
    ])

    # Create MoM chain without memory since we just need to generate once
    chain = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True
    )
    
    return chain

def generate_mom(conversation):
    # Get conversation history from memory
    history = conversation.memory.chat_memory.messages
    
    # Format conversation history as a clear Q&A transcript
    interview_history = ""
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            question = history[i].content
            answer = history[i + 1].content
            interview_history += f"Q: {question}\nA: {answer}\n\n"
    
    # Create and run MoM chain with interview history
    mom_chain = create_mom_chain()
    mom = mom_chain.run(interview_history=interview_history)
    return mom

def main():
    conversation = create_chat()
    interview_completed = False
    
    print("Welcome to the Meeting Analysis Chatbot!")
    print("Type 'hi' to start conversation or Type 'quit' to end the conversation\n")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        
        if user_input == 'quit':
            print("\nThank you for using the Meeting Analysis Chatbot!")
            break
        
        response = conversation.predict(input=user_input)
        print(f"\nBot: {response}")
        
        # Check if interview is complete based on the last bot message
        if not interview_completed:
            last_message = response.lower()
            if ("additional information" in last_message and 
                "would like to add" in last_message):
                interview_completed = True
                print("\nInterview completed! Type 'generate mom' to create Meeting Minutes or continue the conversation.\n")
        
        if user_input == 'generate mom':
            if interview_completed:
                print("\nGenerating Meeting Minutes...\n")
                mom = generate_mom(conversation)
                print("=== Meeting Minutes ===")
                print(mom)
                print("=====================")
            else:
                print("\nPlease complete the interview before generating Meeting Minutes.")
            continue

if __name__ == "__main__":
    main()
