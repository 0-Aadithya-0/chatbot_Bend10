from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr
import torch
import traceback
import random

# Note: The instructions reference pipeline("conversational") and Conversation,
# but personaGPT requires its custom special-token format (<|p2|>, <|sep|>, <|start|>)
# for persona conditioning. We use GPT2LMHeadModel directly to support this.

tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")

# Persona facts — short, simple sentences that personaGPT handles best
persona_facts = [
    "my name is ben tennyson.",
    "people call me ben 10.",
    "i have the omnitrix on my wrist.",
    "i transform into alien heroes.",
    "i am a teenager from bellwood.",
    "i love smoothies and chili fries.",
    "my grandpa max taught me to be a hero.",
    "my cousin gwen has magic powers.",
    "kevin levin is my best friend.",
    "vilgax is my greatest enemy.",
    "i say it's hero time before i transform.",
    "i crack jokes during battles.",
    "heatblast shoots fire.",
    "xlr8 has super speed.",
    "fourarms has super strength.",
    "diamondhead is made of crystal.",
]

# Few-shot seed dialog — teaches the model Ben's tone and personality
fewshot_dialog = [
    ("hi there, who are you?",
     "hey! i'm ben tennyson, the kid with the omnitrix. i turn into awesome aliens to save the world!"),
    ("what do you like to do for fun?",
     "dude, smoothies from mr smoothy are the best. when i'm not saving the world i play sumo slammers or hang out with kevin and gwen."),
    ("tell me about your coolest alien.",
     "oh man, heatblast was my first alien ever, he shoots fire! fourarms is awesome too, four giant arms smashing bad guys!"),
]

# Pre-encode everything once at startup
eos = tokenizer.eos_token
persona_text = "<|p2|>" + eos.join(persona_facts) + eos + "<|sep|>" + "<|start|>"
persona_tokens = tokenizer.encode(persona_text)

fewshot_tokens = []
for user_msg, bot_msg in fewshot_dialog:
    fewshot_tokens.extend(tokenizer.encode(user_msg + eos))
    fewshot_tokens.extend(tokenizer.encode(bot_msg + eos))


def generate_response(dialog_history_tokens):
    """Build the full input sequence and generate a response."""
    input_ids = persona_tokens + fewshot_tokens + dialog_history_tokens

    # Left-truncate history (preserve persona + fewshot) if exceeding context limit
    prefix_len = len(persona_tokens) + len(fewshot_tokens)
    max_input_length = 900  # model max is 1024, leave room for generation
    if len(input_ids) > max_input_length:
        overflow = len(input_ids) - max_input_length
        input_ids = persona_tokens + fewshot_tokens + dialog_history_tokens[overflow:]

    input_tensor = torch.LongTensor([input_ids])
    attention_mask = torch.ones_like(input_tensor)

    output_ids = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=True,
        top_k=10,
        top_p=0.92,
        temperature=0.75,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Extract only newly generated tokens
    response_ids = output_ids[0][len(input_ids):].tolist()

    # Stop at the first eos token if present
    if tokenizer.eos_token_id in response_ids:
        response_ids = response_ids[:response_ids.index(tokenizer.eos_token_id)]

    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def extract_text(content):
    """Extract plain text from Gradio content (handles str, list, and dict formats)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(content)


def vanilla_chatbot(message, history):
    """Gradio ChatInterface callback."""
    try:
        dialog_history_tokens = []

        # Handle both Gradio history formats (dict-style and legacy tuple-style)
        if history and isinstance(history[0], dict):
            for turn in history:
                text = extract_text(turn["content"]) + tokenizer.eos_token
                dialog_history_tokens.extend(tokenizer.encode(text))
        elif history and isinstance(history[0], (list, tuple)):
            for user_msg, bot_msg in history:
                dialog_history_tokens.extend(tokenizer.encode(str(user_msg) + tokenizer.eos_token))
                if bot_msg:
                    dialog_history_tokens.extend(tokenizer.encode(str(bot_msg) + tokenizer.eos_token))

        # Add current user message
        message_text = extract_text(message)
        dialog_history_tokens.extend(tokenizer.encode(message_text + tokenizer.eos_token))

        response = generate_response(dialog_history_tokens)
        return response
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}", flush=True)
        return f"Error: {type(e).__name__}: {e}\n\nTraceback:\n{error_details}"


demo = gr.ChatInterface(
    fn=vanilla_chatbot,
    title="Ben 10 Chatbot",
    description="Chat with Ben Tennyson! The wielder of the Omnitrix is here to talk about his alien adventures, villains, and life as a teenage hero.",
)

demo.launch()
