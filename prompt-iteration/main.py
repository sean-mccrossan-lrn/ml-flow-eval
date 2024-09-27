import functools
import random
import re
import uuid
import logging
import concurrent.futures
from dotenv import load_dotenv
import os

import streamlit as st
from streamlit_feedback import streamlit_feedback
from langsmith import Client as LangSmithClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain import hub
from langchainhub import Client as HubClient


load_dotenv()
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "translation-critic-ds"
PROMPT_NAME = "translation-prompt-few-shot"
OPTIMIZER_PROMPT_NAME = "translation-optimiser"
NUM_FEWSHOTS = 15
PROMPT_UPDATE_BATCHSIZE = 5

st.set_page_config(
    page_title="🛠️ Prompt Refining Tool with Feedback",
    page_icon="🛠️",
    layout="wide",
)

# Sidebar Information
with st.sidebar:
    st.title("Session Information")
    st.markdown("Configure your session settings below.")

    prompt_version = st.text_input(
        "Prompt Version",
        value="latest",
        help="Specify the version of the prompt you want to use."
    )

    chat_llm_model = st.selectbox(
        "Chat LLM Model",
        ["haiku", "opus"],
        help="Choose the language model for the chat assistant."
    )

    prompt_url = f"https://smith.langchain.com/prompts/{PROMPT_NAME}?organizationId=f7030199-7742-4995-9592-ef265539053a"
    if prompt_version and prompt_version != "latest":
        prompt_url = f"{prompt_url}/{prompt_version}"

    optimizer_prompt_url = f"https://smith.langchain.com/prompts/{OPTIMIZER_PROMPT_NAME}?organizationId=f7030199-7742-4995-9592-ef265539053a"

    st.markdown(f"[View Chatbot Prompt]({prompt_url})")
    st.markdown(f"[View Optimizer Prompt]({optimizer_prompt_url})")

# Initialize LLMs
def initialize_llm(model_name):
    if model_name == "haiku":
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    else:
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    return ChatBedrock(
        region_name="us-east-1",
        credentials_profile_name="learnosity-ds-us",
        model_id=model_id,
        provider="anthropic",
        model_kwargs={"temperature": 0.0},
    )

chat_llm = initialize_llm(chat_llm_model)
optimizer_llm = initialize_llm("optimizer_model") 

# Initialize LangSmith client
client = LangSmithClient()

def _format_example(example):
    return f"""<example>
<original>
{example.inputs['input']}
</original>
<refined>
{example.outputs['output']}
</refined>
</example>"""

def few_shot_examples():
    if client.has_dataset(dataset_name=DATASET_NAME):
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        if not examples:
            return ""
        examples = random.sample(examples, min(len(examples), NUM_FEWSHOTS))
        e_str = "\n".join([_format_example(e) for e in examples])
        return f"Approved Examples:\n{e_str}"
    return ""

if 'few_shots' not in st.session_state:
    st.session_state['few_shots'] = few_shot_examples()
few_shots = st.session_state['few_shots']

prompt: ChatPromptTemplate = hub.pull(
    PROMPT_NAME
    + (f":{prompt_version}" if prompt_version and prompt_version != "latest" else "")
)

prompt = prompt.partial(examples=few_shots)

prompt_refiner = (prompt | chat_llm | StrOutputParser()).with_config(run_name="Chat Bot")

def parse_refined_output(response: str, turn: int, box=None):
    match = re.search(r"(.*?)<refined>(.*?)</refined>(.*?)", response.strip(), re.DOTALL)
    box = box or st
    pre, refined_output, post = match.groups() if match else (response, None, None)
    if pre:
        box.markdown(pre)
    if refined_output is not None:
        refined_output = box.text_area(
            "Edit this to save your refined output:",
            refined_output.strip(),
            key=f"refined_{turn}",
            height=300,
        )
    if post:
        box.markdown(post)
    return refined_output

def log_feedback(
    value: dict,
    *args,
    presigned_url: str,
    original_input: str,
    refined_text: str,
    **kwargs,
):
    st.session_state["session_ended"] = True
    st.chat_message("system").markdown("🙏 **Thank you for your feedback! Updating the prompt based on your input...**")

    st.chat_message("system").markdown("🔄 **Starting prompt optimization...**")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        score = {"👍": 1, "👎": 0}.get(value["score"], 0)
        comment = value.get("text", "")
        futures.append(
            executor.submit(
                client.create_feedback_from_token,
                presigned_url,
                score=int(score),
                comment=comment,
            )
        )

        # Function to create a new example in the dataset
        def create_example():
            logger.info(f"Creating example: {original_input} -> {refined_text}")
            try:
                client.create_example(
                    inputs={"input": original_input},
                    outputs={"output": refined_text},
                    dataset_name=DATASET_NAME,
                )
                st.chat_message("system").markdown("💾 **Example saved successfully.**")
            except Exception as e:
                logger.warning(f"Failed to save example: {e}")
                client.create_dataset(dataset_name=DATASET_NAME)
                client.create_example(
                    inputs={"input": original_input},
                    outputs={"output": refined_text},
                    dataset_name=DATASET_NAME,
                )
                st.chat_message("system").markdown("💾 **Example saved successfully after creating dataset.**")

        if score and original_input and refined_text:
            logger.info("Saving example based on positive feedback.")
            futures.append(executor.submit(create_example))
        else:
            logger.info("Example not saved due to negative feedback or missing data.")
            st.chat_message("system").markdown("ℹ️ **Example not saved due to negative feedback or missing data.**")

        with st.spinner("Optimizing prompt..."):
            st.chat_message("system").markdown("📥 **Fetching prompt versions for optimization...**")

            def parse_updated_prompt(system_prompt_txt: str):
                return (
                    system_prompt_txt.split("<improved_prompt>")[1]
                    .split("</improved_prompt>")[0]
                    .strip()
                )

            def format_conversation(messages: list):
                tmpl = """<turn idx="{i}">
{role}: {txt}
</turn idx="{i}">"""
                return "\n".join(
                    tmpl.format(i=i, role=msg[0], txt=msg[1])
                    for i, msg in enumerate(messages)
                )

            hub_client = HubClient()

            def pull_prompt(hash_):
                prompt = hub.pull(f"{PROMPT_NAME}:{hash_}")
                logger.debug(f"Pulled prompt for hash {hash_}: {prompt}")
                return prompt

            def get_prompt_template(prompt):
                if hasattr(prompt, 'messages') and prompt.messages and len(prompt.messages) > 0:
                    system_message = prompt.messages[0]
                    if hasattr(system_message, 'prompt') and hasattr(system_message.prompt, 'template'):
                        return system_message.prompt.template
                    else:
                        raise AttributeError("The first message does not contain a 'prompt' or 'template' attribute.")
                else:
                    logger.error("Prompt messages are empty.")
                    raise IndexError("The 'prompt.messages' list is empty or not properly defined.")

            optimizer_prompt_future = executor.submit(hub.pull, OPTIMIZER_PROMPT_NAME)
            list_response = client.pull_prompt_commit(PROMPT_NAME)

            if isinstance(list_response, list):
                latest_commits = list_response
            elif hasattr(list_response, 'commit_hash'):
                latest_commits = [list_response]
            else:
                latest_commits = []

            if not latest_commits:
                logger.error("No commits found for the prompt.")
                st.chat_message("system").markdown("❌ **Cannot update prompt because no previous versions are available.**")
                return
            hashes = [commit.commit_hash for commit in latest_commits]

            prompt_futures = [executor.submit(pull_prompt, hash_) for hash_ in hashes]
            updated_prompts = [future.result() for future in prompt_futures]
            optimizer_prompt = optimizer_prompt_future.result()

            st.chat_message("system").markdown(f"📄 **Fetched {len(updated_prompts)} prompt versions for optimization.**")

            for idx, (hash_, updated_prompt) in enumerate(zip(hashes, updated_prompts)):
                prompt_template = get_prompt_template(updated_prompt)
                st.chat_message("system").markdown(f"### Prompt Version {idx+1} (Hash: {hash_})\n```\n{prompt_template}\n```")

            optimizer = (
                optimizer_prompt
                | optimizer_llm
                | StrOutputParser()
                | parse_updated_prompt
            ).with_config(run_name="Optimizer")

            try:
                logger.info("Updating the system prompt using optimizer.")
                conversation = format_conversation(
                    st.session_state.get("langchain_messages", [])
                )
                if score:
                    conversation = f'<rating>User rated the conversation as {value["score"]}.</rating>\n\n{conversation}'

                if not hashes or not updated_prompts:
                    logger.error("No hashes or updated prompts available for optimization.")
                    st.chat_message("system").markdown("❌ **Cannot update prompt because no previous versions are available.**")
                    return

                prompt_versions = "\n\n".join(
                    [
                        f'<prompt version="{hash_}">\n{get_prompt_template(updated_prompt)}\n</prompt>'
                        for hash_, updated_prompt in zip(hashes, updated_prompts)
                    ]
                )

                st.chat_message("system").markdown(f"📝 **Data sent to optimizer:**\n\n**Prompt Versions:**\n```\n{prompt_versions}\n```\n**Current Prompt:**\n```\n{get_prompt_template(prompt)}\n```\n**Conversation:**\n```\n{conversation}\n```\n**Final Value:**\n```\n{refined_text}\n```")

                updated_sys_prompt = optimizer.invoke(
                    {
                        "prompt_versions": prompt_versions,
                        "current_prompt": get_prompt_template(prompt),
                        "conversation": conversation,
                        "final_value": refined_text,
                    }
                )

                st.chat_message("system").markdown(f"✨ **Optimized Prompt:**\n```\n{updated_sys_prompt}\n```")

                updated_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", updated_sys_prompt),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                hub.push(PROMPT_NAME, updated_prompt)

                st.chat_message("system").markdown("✅ **Prompt updated successfully!**")

            except Exception as e:
                logger.warning(f"Failed to update prompt: {e}")
                st.chat_message("system").markdown("❌ **An error occurred while updating the prompt.**")

            concurrent.futures.wait(futures)

if 'langchain_messages' not in st.session_state:
    st.session_state['langchain_messages'] = []
messages = st.session_state['langchain_messages']
original_input = messages[0][1] if messages else None

# Main content
st.markdown("## Welcome to the Prompt Refining Tool!")
st.write("Enter your data or prompt below to get started.")

for i, msg in enumerate(messages):
    role, content = msg[0], msg[1]
    if role == 'user':
        st.chat_message("user").write(content)
    elif role == 'assistant':
        if i == len(messages) - 1 and not st.session_state.get("session_ended", False):
            continue
        st.chat_message("assistant").write(content)
    elif role == 'system':
        st.chat_message("system").markdown(content)

if messages and messages[-1][0] == 'assistant' and len(messages[-1]) == 3 and not st.session_state.get("feedback_submitted", False):
    refined_text = parse_refined_output(messages[-1][1], len(messages)-1)
    presigned_url = messages[-1][2]

    feedback = streamlit_feedback(
        feedback_type="thumbs",
        on_submit=functools.partial(
            log_feedback,
            presigned_url=presigned_url,
            original_input=original_input,
            refined_text=refined_text,
        ),
        key=f"fb_{len(messages)-1}",
    )

run_id = uuid.uuid4()
presigned = client.create_presigned_feedback_token(
    run_id, feedback_key="prompt_refinement_quality"
)

if st.session_state.get("session_ended"):
    st.write("Session ended. You can copy your refined output above.")
    if st.button("Start New Session"):
        st.session_state.clear()
        st.rerun()
else:
    if user_input := st.chat_input(placeholder="Type your data here..."):
        st.chat_message("user").write(user_input)
        original_input = user_input
        messages.append(("user", user_input))

        st.chat_message("system").markdown("🤖 **Processing your input...**")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Processing..."):
                full_response = ""
                write_stream = prompt_refiner.stream(
                    {"messages": [tuple(msg[:2]) for msg in messages]},
                    config={"run_id": run_id},
                )
                for chunk in write_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)


        refined_text = parse_refined_output(
            full_response, len(messages)
        )

        messages.append(("assistant", full_response, presigned.url))
        st.session_state["langchain_messages"] = messages

        st.chat_message("system").markdown("📊 **Awaiting your feedback...**")

        feedback = streamlit_feedback(
            feedback_type="thumbs",
            on_submit=functools.partial(
                log_feedback,
                presigned_url=presigned.url,
                original_input=original_input,
                refined_text=refined_text,
            ),
            key=f"fb_{len(messages) - 1}",
        )

# Footer
st.markdown("""
---
*Developed by Learnosity AiLabs*
""")
