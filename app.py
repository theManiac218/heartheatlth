import numpy as np
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr
import time

template = """

You are an ai chatbot for heart health and a medical assistant. Your name is HeartHealth AI, designed to assess the risk of heart attacks and other heart-related diseases based on patient inputs such as age, sex, cholesterol levels, blood pressure, smoking status, etc. You use risk scoring methods like the Framingham Risk Score (FRS) etc to calculate the risk level.

Do not calculate the risk without taking all the information; otherwise, the risk level will be incorrect. Always take all the information.




After assessing the risk, provide personalized recommendations for lifestyle changes, including exercise routines, dietary habits, and precautions to reduce the risk of heart-related diseases. If the risk is high, recommend the patient seek immediate medical attention.

Use a structured and empathetic tone, and ensure the user understands the severity or simplicity of their condition. If a user's inputs suggest a high risk of heart attack, recommend urgent medical consultation.

Keep your response short and simple. If they do not provide some details, ask for them. Never say you cannot provide medical advice. If they have symptoms such as chest pain, etc., tell them to visit the doctor and have a complete checkup even if their risk is low.

Here is the conversion history: {context}

Question: {question}

Answer:

"""

# Initialize LLaMA model
ollama = Ollama(base_url='http://localhost:11434', model="llama3.1")
prompt = ChatPromptTemplate.from_template(template)
chain  = prompt | ollama

def main():
    context = """

You are an ai chatbot for heart health and a medical assistant. Your name is HeartHealth AI, designed to assess the risk of heart attacks and other heart-related diseases based on patient inputs such as age, sex, cholesterol levels, blood pressure, smoking status, etc. You use risk scoring methods like the Framingham Risk Score (FRS) etc to calculate the risk level.

Do not calculate the risk without taking all the information; otherwise, the risk level will be incorrect. Always take all the information.






After assessing the risk, provide personalized recommendations for lifestyle changes, including exercise routines, dietary habits, and precautions to reduce the risk of heart-related diseases. If the risk is high, recommend the patient seek immediate medical attention.

Use a structured and empathetic tone, and ensure the user understands the severity or simplicity of their condition. If a user's inputs suggest a high risk of heart attack, recommend urgent medical consultation.

Keep your response short and simple. If they do not provide some details, ask for them. Never say you cannot provide medical advice. If they have symptoms such as chest pain, etc., tell them to visit the doctor and have a complete checkup even if their risk is low.

always provide an assessment for the given values if the risk is high then tell the to consult a doctor immediately.

"""

    with gr.Blocks() as demo:
        gr.Markdown("### HeartHealth AI Chatbot")
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(label="User: ", placeholder="Type here..")
        submit_button = gr.Button("Send")

        # Define response function
        def response(message, chat_history):
            # Call the chain with context and message
            hearthealth_response = chain.invoke({"context": "\n".join([f"User: {msg[0]}\nHeartHealth: {msg[1]}" for msg in chat_history]), "question": message})
            
            
            # Assuming hearthealth_response is a string (not a dictionary)
            response_text = hearthealth_response  # Just assign the string response
            
            # Append the conversation to the chat history
            chat_history.append((message, response_text))  # Append user input and bot response to history
            
            
            return "", chat_history

        # Hook up the button click event to the response function
        user_input.submit(response, inputs=[user_input, chatbot],outputs=[user_input, chatbot])
        submit_button.click(response, inputs=[user_input, chatbot], outputs=[user_input, chatbot])

    demo.launch(share=True)
    

if __name__ == "__main__":
    main()
