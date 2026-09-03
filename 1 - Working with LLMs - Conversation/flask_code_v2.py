from flask import Flask, request, render_template, redirect, url_for
from openai import OpenAI, AuthenticationError, RateLimitError
import os

from dotenv import load_dotenv

load_dotenv(
    "/Users/shivam13juna/Documents/scaler/GEN_AI_REF/openai_key.env"
)  # reads .env file in the current directory


import truststore

truststore.inject_into_ssl()


MODEL = "gpt-5-nano"
SYSTEM_PROMPT = (
    "You are a friendly Python tutor who explains concepts clearly. "
    "Keep paragraphs short and always put code inside fenced markdown blocks (```python)."
)

# templates_v2 keeps this app's UI separate from the original flask_code.py
app = Flask(__name__, template_folder="templates_v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Shared conversation state (for demo only, reset on server restart)
messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # what we send to the API
history = []  # what the page renders: {"role", "content", "usage"}
totals = {"prompt": 0, "completion": 0, "total": 0}


def safe_chat_call(messages):
    try:
        return client.chat.completions.create(model=MODEL, messages=messages), None
    except AuthenticationError:
        return None, "Authentication failed — check the OPENAI_API_KEY in your env file."
    except RateLimitError:
        return None, "Rate limit hit — wait a few seconds and send it again."
    except Exception as e:
        return None, f"Something went wrong: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form.get("user_input", "").strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
            history.append({"role": "user", "content": user_text, "usage": None})

            resp, error = safe_chat_call(messages)
            if resp:
                reply = resp.choices[0].message.content
                messages.append({"role": "assistant", "content": reply})

                usage = resp.usage
                # gpt-5 models hide reasoning tokens inside completion_tokens
                details = getattr(usage, "completion_tokens_details", None)
                reasoning = getattr(details, "reasoning_tokens", 0) or 0
                history.append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "usage": {
                            "prompt": usage.prompt_tokens,
                            "completion": usage.completion_tokens,
                            "reasoning": reasoning,
                            "total": usage.total_tokens,
                        },
                    }
                )
                totals["prompt"] += usage.prompt_tokens
                totals["completion"] += usage.completion_tokens
                totals["total"] += usage.total_tokens
            else:
                history.append({"role": "error", "content": error, "usage": None})

        # POST -> redirect -> GET, so a browser refresh doesn't resend the message
        return redirect(url_for("index"))

    return render_template("index.html", history=history, totals=totals, model=MODEL)


@app.route("/reset", methods=["POST"])
def reset():
    del messages[1:]  # keep the system prompt
    history.clear()
    totals.update(prompt=0, completion=0, total=0)
    return redirect(url_for("index"))


# command to run flask app

# flask run --app flask_code_v2.py --debug
