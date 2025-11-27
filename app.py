import os
from flask import Flask, request, Response, stream_with_context
from ollama import Client

app = Flask(__name__)

# Initialize Client
client = Client(
    host=os.environ.get('OLLAMA_HOST', 'https://ollama.com'),
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

# Global dictionary to store history
# Format: { 'user_id': [ {'role': 'user', ...}, {'role': 'assistant', ...} ] }
chat_store = {}

@app.route('/content', methods=['GET'])
def generate_content():
    user_prompt = request.args.get('content')
    # Get a session ID (defaults to 'default' if not provided)
    user_id = request.args.get('id', 'default')

    if not user_prompt:
        return "Error: No content parameter provided.", 400

    # Initialize history for this user if it doesn't exist
    if user_id not in chat_store:
        chat_store[user_id] = []

    # Add the new user message to history
    chat_store[user_id].append({'role': 'user', 'content': user_prompt})

    def generate():
        full_response = ""
        try:
            # Send the FULL history to Ollama
            stream = client.chat(
                model='gpt-oss:120b', 
                messages=chat_store[user_id], 
                stream=True
            )
            
            for part in stream:
                chunk = part['message']['content']
                full_response += chunk # Accumulate the text
                yield chunk
            
            # Once stream finishes, save the assistant's reply to history
            chat_store[user_id].append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
