# Your code goes here
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

prompt_template_str = """
Your task is to explain the concept of **{concept}** to me in a way that is:

1. Clear and intuitive
2. Concise (in under 100 words)
3. Tailored specifically to me and what I already know

Use the following information about me to personalize your explanation:

- Background: Aspiring AI/Data Engineer with Python skills
- Professional Interests: AI Engineering, Data Engineering, Forward Deployment Engineering
- Domain Focus: Healthcare and Mining industries
- Goal: Understanding concepts that help me break into AI/Data Engineering roles

The personalization should be subtle and natural. Avoid forced references to my background that don't genuinely enhance understanding.
"""

# Create a prompt template
prompt_template = PromptTemplate.from_template(prompt_template_str)

# Create a model interface
model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")


def generate_explanation(input_text):
  # Format the prompt with the input variable
  prompt = prompt_template.format(concept=input_text)
  # Call the model with the prompt
  response = model.invoke(prompt)
  return response.content


demo = gr.Interface(
    fn=generate_explanation,
    inputs=[gr.Textbox(label="Enter a concept", lines=1)],
    outputs=[gr.Textbox(label="Explanation", lines=5)],
    flagging_mode="never",
    title="ClarifyAI – Personalized Concept Explainer",
    description="Enter any term and get a concise explanation tailored to your background"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
