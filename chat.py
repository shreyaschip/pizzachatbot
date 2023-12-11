import streamlit as st
import replicate
import os
import pandas as pd

try:
    # Assuming your dataset is a CSV file with columns 'user_prompt' and 'assistant_response'
    custom_dataset_path = r".\pizza.csv"
    custom_dataset_data = pd.read_csv(custom_dataset_path)
    custom_dataset = pd.DataFrame(custom_dataset_data)

    print(type(custom_dataset))
    print(custom_dataset.info())

    #custom_dataset = pd.DataFrame(custom_dataset_data)
    #print(custom_dataset.head())
    # App title
    st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Llama 2 Chatbot')
        if 'REPLICATE_API_TOKEN' in st.secrets:
            st.success('API key already provided!', icon='✅')
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api

        st.subheader('Models and parameters')
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
        if selected_model == 'Llama2-7B':
            llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
        elif selected_model == 'Llama2-13B':
            llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
        st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    def generate_llama2_response(prompt_input):
        global custom_dataset
        # Use the custom dataset if the prompt matches any user prompt in the dataset
        user_prompt_match = custom_dataset[custom_dataset['user_prompt'] == prompt_input]
        print(f"prompt_input: {prompt_input}")
        print(f"user_prompt_match: {user_prompt_match}")

        if not user_prompt_match.empty:
            response = user_prompt_match['assistant_response'].values[0]
        else:
            # If no match is found, use the Llama 2 model to generate a response
            string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            for dict_message in st.session_state.messages:
                if dict_message["role"] == "user":
                    string_dialogue += "User: " + dict_message["content"] + "\n\n"
                else:
                    string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

            # Use Llama 2 model to generate a response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
                                input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                        "temperature": temperature, "top_p": top_p, "max_length": max_length, "repetition_penalty": 1})
            response = ''.join(output)

            # Append the new user prompt and assistant response to the custom dataset
            custom_dataset = pd.concat([custom_dataset, pd.DataFrame({'user_prompt': [prompt_input], 'assistant_response': [response]})], ignore_index=True)
            custom_dataset.to_csv(custom_dataset_path, index=False)

        return response

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
except Exception as e:
    print(f"An exception occurred: {e}")