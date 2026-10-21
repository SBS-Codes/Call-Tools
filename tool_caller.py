import json
import ollama

# 1. Define your custom tool function
def calculate_hypotenuse(a, b) -> float:
    """Calculates the hypotenuse of a right-angled triangle given sides a and b."""
    # Explicitly cast to float to prevent TypeError if the LLM sends strings
    a_float = float(a)
    b_float = float(b)
    
    print(f"\n[Tool Execution] Running calculate_hypotenuse with a={a_float}, b={b_float}")
    return (a_float**2 + b_float**2) ** 0.5


def calculate_product(a, b) -> float:
    """Calculates the product of two numbers."""
    a_float = float(a)
    b_float = float(b)

    print(f"\n[Tool Execution] Running calculate_product with a={a_float}, b={b_float}")
    return a_float * b_float


# 2. Map your python function to a tool definition list
# You must specify types and descriptions so Llama 3.2 knows how to use it.
my_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'calculate_hypotenuse',
            'description': 'Calculate the hypotenuse of a right-angled triangle.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'number', 'description': 'Length of side a'},
                    'b': {'type': 'number', 'description': 'Length of side b'},
                },
                'required': ['a', 'b'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_product',
            'description': 'Calculate the product of two numbers.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'number', 'description': 'First number'},
                    'b': {'type': 'number', 'description': 'Second number'},
                },
                'required': ['a', 'b'],
            },
        },
    },
]

# A dictionary to quickly route string names to actual Python functions
available_tools = {
    'calculate_hypotenuse': calculate_hypotenuse,
    'calculate_product': calculate_product,
}

def run_conversation():
    # The prompt requiring a tool call
    # user_prompt = "I have a right triangle where side A is 3 cm and side B is 4 cm. What is the hypotenuse?"
    user_prompt = "I have two numbers A and B. A is 3 and B is 4. What is the product?"
    
    messages = [{'role': 'user', 'content': user_prompt}]
    print(f"User Prompt: {user_prompt}")

    # 3. First LLM Call: Pass the prompt and the available tools
    response = ollama.chat(
        model='llama:latest',
        messages=messages,
        tools=my_tools,  # Inform the model about your tool
    )

    # Append the model's response to the conversation history
    messages.append(response['message'])

    # 4. Check if the model decided to call a tool
    if response['message'].get('tool_calls'):
        for tool_call in response['message']['tool_calls']:
            tool_name = tool_call['function']['name']
            arguments = tool_call['function']['arguments']
            
            if tool_name in available_tools:
                # Execute the actual Python tool function with the model's generated inputs
                tool_output = available_tools[tool_name](**arguments)
                
                # 5. Send the tool results back to the model
                messages.append({
                    'role': 'tool',
                    'name': tool_name,
                    'content': json.dumps({'result': tool_output}),
                })
                
                # Second LLM Call: Model reads the tool output and drafts the final answer
                final_response = ollama.chat(
                    model='llama:latest',
                    messages=messages
                )
                print(f"\nLlama Final Answer:\n{final_response['message']['content']}")
    else:
        # Fallback if the model didn't think it needed a tool
        print(f"\nLlama Final Answer:\n{response['message']['content']}")

if __name__ == "__main__":
    run_conversation()
