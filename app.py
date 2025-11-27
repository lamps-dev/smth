import os
from flask import Flask, request, Response, stream_with_context
from ollama import Client

app = Flask(__name__)

# Initialize Client
# Ensure OLLAMA_HOST and OLLAMA_API_KEY are set in Render Environment Variables
client = Client(
    host=os.environ.get('OLLAMA_HOST', 'https://ollama.com'),
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

@app.route('/content', methods=['GET'])
def generate_content():
    # Get the prompt from the URL parameter (e.g., ?content=hello)
    user_prompt = request.args.get('content')

    if not user_prompt:
        return "Error: No content parameter provided.", 400

    messages = [
        {'role': 'user', 'content': user_prompt},
    ]

    # function to yield chunks of data for streaming
    def generate():
        try:
            stream = client.chat(model='gpt-oss:120b', messages=messages, stream=True)
            for part in stream:
                yield part['message']['content']
        except Exception as e:
            yield f"Error generating response: {str(e)}"

    # Return a streaming response
    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == '__main__':
    # This is for running locally
    app.run(debug=True)
