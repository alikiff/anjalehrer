import base64
from dotenv import load_dotenv
from pathlib import Path
import os
from mistralai import Mistral, OCRResponse
from typing import List, Dict
import json

def get_mistral_api_key_from_dot_env(path_dot_env: Path) -> str:
    load_dotenv(path_dot_env)
    return os.getenv('MISTRAL_API_KEY')

def get_mistral_client(api_key: str) -> Mistral:
    client = Mistral(api_key=api_key)
    return client

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None
    


def ocr_get_text(mistral_client: Mistral, image_encoded: bytes) -> OCRResponse:
    ocr_response = mistral_client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image_encoded}" 
    }
    )
    markdown_from_all_pages = "".join([page.markdown for page in ocr_response.pages])
    return markdown_from_all_pages


def llm_helper_create_messages(ocr_response_markdown: str) -> List[Dict]:
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
                    <AUFGABE>
                    Der nachfolgende Text wurde aus einem englischen Schulbuch kopiert und in
                    Markdown konvertiert. Er enthält eine 
                    Aufstellung/Liste von englischen Vokabeln, bzw. Phrasen, welche auch mehrere Wörter lang sein können.
                    In der Regel finden sich die Vokabeln in Tabellenform.
                    Detail-Anweisungen dazu:
                    - Wähle aus diesen bitte 10 aus und gibt sie im JSON-FORMAT (wichtig!) zurück.
                    - Denke dir keine Vokabeln aus, verwende nur die, die du als solche in dem Text erkennst.
                    - Findest du weniger als 10, gib entsprechend weniger zurück. Fülle nicht mit 
                      aus gedachten Vokabeln aus!
                    - wenn die Vokabeln aus mehreren Wörtern bestehen, also wie z.B. 'wear a school uniform' bestehen,
                      trenne sie nicht, sondern lasse die Phrase als Vokabel in den Output.

                    <JSON-ZIEL-FORMAT>
                    {{vocabulary_lst: [{{english_vocabulary_word_or_phrase1: translation_to_german1}},
                                       {{english_vocabulary_word_or_phrase2: translation_to_german2}}]}}

                    <TEXT MIT ENTHALTENEN VOKABELN>
                    {ocr_response_markdown}
                    """
            },
        ]
    }]
    return messages


def llm_extract_vocabulary(mistral_client: Mistral, messages: List[Dict]) -> str:
    # Get the chat response
    chat_response = mistral_client.chat.complete(
    model="mistral-large-latest",
    messages=messages,
    response_format={"type": "json_object"},
    temperature=0.2
    )
    print(chat_response)
    return chat_response.choices[0].message.content


def format_get_vocabulary_list(llm_output: str) -> List[Dict]:
    vocabulary_json = json.loads(llm_output.replace("\n", ""))
    print(vocabulary_json)

    assert "vocabulary_lst" in vocabulary_json, "Auslesen der Vokabeln fehlgeschlagen!!!"

    vocabulary_lst = [el for el in vocabulary_json["vocabulary_lst"]]
    print(vocabulary_lst)

    return vocabulary_lst