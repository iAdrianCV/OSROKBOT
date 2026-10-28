import json
import os
import csv
import time
from dotenv import load_dotenv

from Actions.action import Action
from Actions.manual_click_action import ManualClickAction
from Actions.manual_move_action import ManualMoveAction
from Actions.check_color_action import CheckColorAction
from global_vars import GlobalVars

try:
    from termcolor import colored
except ImportError:
    def colored(text, color):
        return text

try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
    from typing import Literal

    class QuizAnswer(BaseModel):
        answer: Literal["A", "B", "C", "D"]
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class ChatGPTAction(Action):
    def __init__(self, midterm=False, filepath="string.txt", delay=0, retard=0):
        load_dotenv()
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.openai_key = os.getenv('OPENAI_KEY')
        self.midterm = midterm
        self.delay = delay
        self.retard = retard
        self.message = ""

        # Setup Gemini client if API key is provided
        self.gemini_client = None
        if self.gemini_key and HAS_GENAI:
            self.gemini_client = genai.Client(
                api_key=self.gemini_key,
                http_options=types.HttpOptions(timeout=10000)
            )

        # Legacy OpenAI function schema fallback
        self.functions = [
            {
                "name": "return_option_based_on_prompt",
                "description": "thinking step by step, ignoring the answer options, returns the chosen answer option (A, B, C or D) based on the prompt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"],
                            "description": "The chosen answer option to the question in the prompt.",
                        },
                    },
                    "required": ["answer"],
                },
            }
        ]

    def checkCorrect(self):
        with open('roklyceum.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            time.sleep(0.5)
            if CheckColorAction(40, 48).execute():
                writer.writerow([GlobalVars().Q, GlobalVars().A])
            elif CheckColorAction(60, 50).execute():
                writer.writerow([GlobalVars().Q, GlobalVars().B])
            elif CheckColorAction(40, 58).execute():
                writer.writerow([GlobalVars().Q, GlobalVars().C])
            elif CheckColorAction(60, 58).execute():
                writer.writerow([GlobalVars().Q, GlobalVars().D])

    def query_gemini(self, prompt):
        models_to_try = ["gemini-3.8-flash", "gemini-flash-latest"]
        for model_name in models_to_try:
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a quiz assistant for the game Rise of Kingdoms. Analyze the question and answer choices, and return the correct option letter (A, B, C, or D).",
                        response_mime_type="application/json",
                        response_schema=QuizAnswer,
                        temperature=0.1
                    )
                )
                data = json.loads(response.text)
                return data.get("answer")
            except Exception as e:
                print(f"[Gemini] Model {model_name} warning: {e}")
                time.sleep(1)
        return None

    def query_openai(self, prompt):
        try:
            import openai
            openai.api_key = self.openai_key
            messages = [
                {"role": "system", "content": "You are a quizz assistant in the game Rise of Kingdoms."},
                {"role": "user", "content": prompt}
            ]
            chat = openai.chat.completions.create(
                model="gpt-4-1106-preview",
                temperature=0.1,
                messages=messages,
                functions=self.functions,
                function_call={"name": "return_option_based_on_prompt"}
            )
            function_arguments = json.loads(chat.choices[0].message.function_call.arguments)
            return function_arguments.get("answer")
        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return None

    def execute(self):
        self.message = (
            f"Question: {GlobalVars().Q}\n"
            f"A: {GlobalVars().A}\n"
            f"B: {GlobalVars().B}\n"
            f"C: {GlobalVars().C}\n"
            f"D: {GlobalVars().D}"
        )
        print("\n\n" + self.message)

        function_response = None

        # 1. Try Gemini first if configured
        if self.gemini_client:
            print("[OSROKBOT] Consultando a Gemini 3.8 Flash...")
            function_response = self.query_gemini(self.message)

        # 2. Fallback to OpenAI if Gemini was not used or failed
        if not function_response and self.openai_key:
            print("[OSROKBOT] Fallback a OpenAI...")
            function_response = self.query_openai(self.message)

        if not function_response:
            print(colored("No se pudo obtener una respuesta de la IA.", "red"))
            return False

        print("\n\n I think it's ", colored(function_response, "red"))

        # Switch case for reply A, B, C, or D
        if not self.midterm:
            if function_response == "A":
                ManualClickAction(40, 48).execute()
            elif function_response == "B":
                ManualClickAction(60, 50).execute()
            elif function_response == "C":
                ManualClickAction(40, 58).execute()
            elif function_response == "D":
                ManualClickAction(60, 58).execute()
            self.checkCorrect()
        else:
            if function_response == "A":
                ManualMoveAction(37, 55).execute()
            elif function_response == "B":
                ManualMoveAction(60, 55).execute()
            elif function_response == "C":
                ManualMoveAction(37, 63).execute()
            elif function_response == "D":
                ManualMoveAction(60, 63).execute()

        return True
