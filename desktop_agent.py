"""
Flow: UI (Tkinter) + Mic -> faster-whisper -> DeepSeek -> Execute Tool -> Speak English (TTS)

Dependencies:
    pip install faster-whisper openai pyttsx3 pillow pycaw comtypes requests speech_recognition pyaudio numpy

Before running:
    Set the environment variable DEEPSEEK_API_KEY, or change API_KEY below.
"""

import os
import json
import time
import threading
import queue
import subprocess
import webbrowser
import urllib.parse
from datetime import datetime

import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import pyttsx3
import speech_recognition as sr
from faster_whisper import WhisperModel
from openai import OpenAI

# ===================== Config =====================
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "API")
BASE_URL = "https://api.deepseek.com"

MODEL = "deepseek-chat"  # deepseek-chat / deepseek-reasoner

WHISPER_SIZE = "large-v3-turbo"  # tiny / base / small / medium / large-v3-turbo
WHISPER_DEVICE = "cuda"  # cpu / cuda
WHISPER_COMPUTE = "float16"  # CPU: int8 / GPU: float16
LANGUAGE = "en"  # Recognize English
# =================================================


# ===================== TTS =====================
def speak(text: str):
    """
    Speaks the text in English.
    Initializing locally prevents the runAndWait() deadlock on Windows.
    """
    engine = pyttsx3.init()
    for _v in engine.getProperty("voices"):
        if "en" in (_v.id + _v.name).lower() or "english" in _v.name.lower():
            engine.setProperty("voice", _v.id)
            break
            
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()


# ===================== Tool Registry =====================
TOOL_SCHEMAS = []
TOOL_FUNCS = {}

def tool(name, description, parameters):
    def decorator(func):
        TOOL_SCHEMAS.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
        TOOL_FUNCS[name] = func
        return func
    return decorator

APP_MAP = {
    "notepad": "notepad",
    "editor": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "paint": "mspaint",
    "cmd": "cmd",
    "terminal": "cmd",
    "chrome": "chrome",
    "edge": "msedge",
    "spotify": "spotify",
}

@tool(
    "open_application",
    "Opens a program on the Windows PC, e.g., notepad, calculator, explorer, paint, browser.",
    {
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "Name of the program"}
        },
        "required": ["app_name"],
    },
)
def open_application(app_name):
    key = app_name.strip().lower()
    if key in ("browser", "web browser", "internet"):
        webbrowser.open("https://www.google.com")
        return "Browser opened."
    target = APP_MAP.get(key, key)
    try:
        subprocess.Popen(target, shell=True)
        return f"{app_name} was opened."
    except Exception as e:
        return f"Could not open {app_name}: {e}"

@tool(
    "web_search",
    "Searches for something on the internet and opens the results in the browser.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
)
def web_search(query):
    webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(query))
    return f"I searched for '{query}'."

@tool(
    "get_current_time",
    "Returns the current date and time.",
    {"type": "object", "properties": {}},
)
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

@tool(
    "get_weather",
    "Fetches the current weather for a city using Open-Meteo.",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "Name of the city, e.g., 'Berlin'"}
        },
        "required": ["city"],
    },
)
def get_weather(city):
    import requests
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en"},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        loc = geo["results"][0]
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current": "temperature_2m,wind_speed_10m"},
            timeout=10,
        ).json()["current"]
        return f"{loc['name']}: {w['temperature_2m']}°C, Wind {w['wind_speed_10m']} km/h."
    except Exception as e:
        return f"Weather request failed: {e}"

@tool(
    "take_screenshot",
    "Takes a screenshot and saves it to the Desktop.",
    {"type": "object", "properties": {}},
)
def take_screenshot():
    try:
        from PIL import ImageGrab
        fname = f"screenshot_{int(time.time())}.png"
        path = os.path.join(os.path.expanduser("~"), "Desktop", fname)
        ImageGrab.grab().save(path)
        return f"Screenshot saved to Desktop: {fname}"
    except Exception as e:
        return f"Screenshot failed: {e}"


# ===================== Agent Main Body =====================
SYSTEM_PROMPT = (
    "You are a helpful voice assistant on a Windows PC. "
    "ALWAYS answer in English, keep it short, conversational, and natural – your answer will be spoken out loud. "
    "If the user wants an action (open app, weather, screenshot, web search), use the appropriate tools. "
    "For normal questions, answer directly without a tool."
)

class DesktopAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def handle_command(self, user_text, ui_callback):
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(5):
            resp = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            self.messages.append(msg)

            if not msg.tool_calls:
                return msg.content

            for tc in msg.tool_calls:
                func = TOOL_FUNCS.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = func(**args) if func else f"Unknown Tool: {tc.function.name}"
                except Exception as e:
                    result = f"Tool Error: {e}"
                
                # Send tool usage info to UI
                ui_callback(f"[System] Used tool: {tc.function.name} -> {result}")
                
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                )

        return "Sorry, that took too many steps."


# ===================== Voice Recognition Thread =====================
class VoiceListener(threading.Thread):
    def __init__(self, callback):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback
        self.daemon = True
        self.running = True
        self.paused = False

        self.callback("[System] Loading Whisper Model...")
        self.model = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        self.callback("[System] Model loaded. Microphone is ready.")

    def run(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = self.model.transcribe(samples, language=LANGUAGE, beam_size=5, vad_filter=True)
                    text = "".join(seg.text for seg in segments).strip()
                    if text:
                        self.callback(text, is_voice=True)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    pass


# ===================== UI & Main Logic =====================
class AssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSeek Voice Assistant")
        self.root.geometry("600x500")
        self.root.configure(bg="#f4f4f9")

        self.queue = queue.Queue()
        self.agent = DesktopAgent()

        # UI Setup
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 11), bg="#ffffff", state=tk.DISABLED)
        self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar()
        self.status_var.set("Status: Initializing...")
        self.status_label = tk.Label(root, textvariable=self.status_var, font=("Segoe UI", 9, "italic"), bg="#f4f4f9", fg="#555")
        self.status_label.pack(anchor="w", padx=10)

        input_frame = tk.Frame(root, bg="#f4f4f9")
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.text_input = tk.Entry(input_frame, font=("Segoe UI", 12))
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.text_input.bind("<Return>", self.on_text_submit)

        self.send_btn = tk.Button(input_frame, text="Send", font=("Segoe UI", 10, "bold"), bg="#0078D7", fg="white", command=self.on_text_submit)
        self.send_btn.pack(side=tk.RIGHT)

        # Start Voice Listener
        self.listener = VoiceListener(self.on_voice_input)
        self.listener.start()

        # Start UI Queue checker
        self.root.after(100, self.process_queue)
        self.append_to_chat("🤖 Assistant: Hello! I am your desktop assistant. You can speak to me or type below.", "assistant")

    def append_to_chat(self, text, role="system"):
        self.chat_display.config(state=tk.NORMAL)
        if role == "user":
            self.chat_display.insert(tk.END, f"👤 You: {text}\n\n", "user")
            self.chat_display.tag_config("user", foreground="#000000", font=("Segoe UI", 11, "bold"))
        elif role == "assistant":
            self.chat_display.insert(tk.END, f"🤖 Assistant: {text}\n\n", "assistant")
            self.chat_display.tag_config("assistant", foreground="#005a9e")
        else:
            self.chat_display.insert(tk.END, f"{text}\n\n", "system")
            self.chat_display.tag_config("system", foreground="#7a7a7a", font=("Segoe UI", 9))
            
        self.chat_display.yview(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def on_voice_input(self, text, is_voice=False):
        if is_voice:
            self.queue.put({"type": "user_input", "text": text})
        else:
            self.queue.put({"type": "system_msg", "text": text})

    def on_text_submit(self, event=None):
        text = self.text_input.get().strip()
        if text:
            self.text_input.delete(0, tk.END)
            self.queue.put({"type": "user_input", "text": text})

    def process_llm_request(self, text):
        self.listener.paused = True  # Pause microphone
        self.queue.put({"type": "status", "text": "Status: Thinking..."})
        
        # Pass a callback to the agent to print tool usage to the GUI
        reply = self.agent.handle_command(text, lambda msg: self.queue.put({"type": "system_msg", "text": msg}))
        
        self.queue.put({"type": "assistant_reply", "text": reply})
        self.queue.put({"type": "status", "text": "Status: Speaking..."})
        
        speak(reply)
        
        self.queue.put({"type": "status", "text": "Status: Listening..."})
        self.listener.paused = False  # Resume microphone

    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg["type"] == "system_msg":
                    self.append_to_chat(msg["text"], "system")
                    if "Microphone is ready" in msg["text"]:
                        self.status_var.set("Status: Listening...")
                
                elif msg["type"] == "user_input":
                    self.append_to_chat(msg["text"], "user")
                    # Start a background thread to call API so UI doesn't freeze
                    threading.Thread(target=self.process_llm_request, args=(msg["text"],), daemon=True).start()
                
                elif msg["type"] == "assistant_reply":
                    self.append_to_chat(msg["text"], "assistant")
                
                elif msg["type"] == "status":
                    self.status_var.set(msg["text"])
                    
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantGUI(root)
    root.mainloop()