import streamlit as st
import os
from app import create_chat, generate_mom


# Set Deepseek API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Openai API key is not set!")
    st.stop()

# Sidebar Guide
with st.sidebar:
    # Generate MoM button is always available
    generate_mom_clicked = st.button("Generate MoM", use_container_width=True, key="sidebar_generate")
    
    st.markdown("---")
    st.header("How to Use")
    st.markdown("""
    1. **Start the Conversation**
        * Type 'hi' to begin the interview
        * The bot will ask you essential questions about the meeting
    
    2. **During the Interview**
        * Answer each question clearly and concisely
        * Provide all relevant details requested
        * The bot will ask follow-up questions as needed
    
    3. **Completing the Interview**
        * The bot will confirm when all necessary information is gathered
        * You can add any additional information if needed
    
    4. **Generating Minutes**
        * Once the interview is complete, click 'Generate Minutes'
        * The bot will create a formatted Meeting Minutes document
    
    5. **End Session**
        * Type 'quit' to end the conversation
    """)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = create_chat()
    st.session_state.messages = []
    st.session_state.interview_completed = False
    # Add welcome messages to session state
    welcome_msg = "Welcome to the Meeting Analysis Chatbot!"
    st.session_state.messages.extend([
        {"role": "assistant", "content": welcome_msg}
    ])

# Title
st.title("Meeting Minutes Bot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Generate MoM button click
if generate_mom_clicked:
    if len(st.session_state.messages) > 1:  # Check if there's any conversation
        with st.spinner("Generating Meeting Minutes..."):
            try:
                mom = generate_mom(st.session_state.conversation)
                # Display MoM in chat
                with st.chat_message("assistant"):
                    st.markdown("### Generated Meeting Minutes")
                    st.markdown(str(mom))  # Ensure MoM is string
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"### Generated Meeting Minutes\n\n{mom}"
                })
                # Force update
                st.rerun()
            except Exception as e:
                error_msg = f"Error generating MoM: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            st.warning("Please start a conversation first before generating minutes.")

# Chat input with unique key
if prompt := st.chat_input("Type your message here...", key="chat_input"):
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        try:
            # Get response from conversation chain
            chain_response = st.session_state.conversation.predict(input=prompt)
            
            # Ensure response is a string
            if isinstance(chain_response, dict):
                response = chain_response.get('response', str(chain_response))
            else:
                response = str(chain_response)
            
            # Display response
            st.markdown(response)
            
            # Add to message history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Force a rerun to update the chat display
    st.rerun()
