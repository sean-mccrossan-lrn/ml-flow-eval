from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import ChatPromptTemplate

def call_bedrock_model_with_langchain(model_config, prompt, data):
    """
    Use Langchain to call AWS Bedrock with the Anthropic Claude model for language translation.
    """
   
    chat = BedrockChat(
        credentials_profile_name="learnosity-ds-us", 
        model_id=model_config.actual_model_name,  
        region_name="us-east-1" 
    )
    system_prompt = prompt

    human_prompt = f"Here's a text to translate: {data}"

    chat_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])

    response = chat_prompt | chat
    response_content = response.invoke({"content": prompt})

    return response_content.content
