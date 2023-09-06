from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# image to text model (BLIP)

def img2txt(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text

#scenario = img2txt("B-780.jpg")


# api to gpt 3.5

def generate_story(scenario):
    template= """
    You are a great fantasy storyteller. You excel at dungeons and dragons;
    You can generate a short story based on a simple narative, the story should not be 
    longer than 80 words;
    
    CONTEXT: {scenario}
    STORY:
    """

    prompt_l = PromptTemplate(template=template, input_variables= ["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name = "gpt-3.5-turbo", temperature=1), prompt=prompt_l, verbose=True
    )

    story= story_llm.predict(scenario=scenario)
    print(story)
    return story

#story = generate_story(scenario)

# text to speech
def text2speech(story):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": story
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

    
#text2speech(story)

def main():
    st.set_page_config(page_title="img to fantasy story")
    # Custom CSS for fantasy background
    custom_css = """
    <style>
        body {
            background-image: url('https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/05d6eeeb-c3a9-44e0-a812-f0c1142d6487/de9urfx-58af5c18-565f-4fc2-9db1-a5a128764b25.jpg/v1/fill/w_1280,h_720,q_75,strp/final_fantasy_background_by_marktailor_de9urfx-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NzIwIiwicGF0aCI6IlwvZlwvMDVkNmVlZWItYzNhOS00NGUwLWE4MTItZjBjMTE0MmQ2NDg3XC9kZTl1cmZ4LTU4YWY1YzE4LTU2NWYtNGZjMi05ZGIxLWE1YTEyODc2NGIyNS5qcGciLCJ3aWR0aCI6Ijw9MTI4MCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.LSUR4zQuriU3mIAwj8nYM-_RiyyZp_LeYhkWbRK4S10');
            background-size: cover;
        }
        .stApp {
            opacity: 0.9;
        }
    </style>
    """
    
    st.markdown(custom_css, unsafe_allow_html=True)
    
    

    st.header("Turn image into a fantasy audio story")
    uploaded_file = st.file_uploader("Choose an image..", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image",
                  use_column_width=True)
        
        # Add a button for generating the story
        if st.button("Generate"):
            scenario = img2txt(uploaded_file.name)
            story = generate_story(scenario)
            text2speech(story)

            with st.expander("scenario"):
                st.write(scenario)
            with st.expander("story"):
                st.write(story)
            
            st.audio("audio.flac")

if __name__ == "__main__":
    main()

