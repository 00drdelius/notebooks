from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function
from openai.types.completion_usage import CompletionUsage


MESSAGE1=[
    {'role': 'system', 'content': '你是一名人工智能助手，你擅长使用工具回答用户问题。'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."}
]

RESPONSE1=ChatCompletion(
    id='chatcmpl-f358f1275d3346d6b9fbf88b962c1c94',
    choices=[
        Choice(
            finish_reason='tool_calls',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=None,
                refusal=None,
                role='assistant',
                annotations=None,
                audio=None,
                function_call=None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id='chatcmpl-tool-5e2ec27d2c0443ff919b058f8d4f1f9f',
                        function=Function(arguments='{"location": "San Francisco, California, USA"}', name='get_current_temperature'),
                        type='function'
                    ),
                    ChatCompletionMessageFunctionToolCall(
                        id='chatcmpl-tool-ce05bda54780485da364b820ae9f5a2f',
                        function=Function(arguments='{"location": "San Francisco, California, USA", "date": "2024-10-01"}', name='get_temperature_date'),
                        type='function'
                    )
                ],
                reasoning_content=None
            ),
            stop_reason=None,
            token_ids=None
        )
    ],
    created=1770971416,
    model='Qwen3-30B-A3B-Instruct-2507-2',
    object='chat.completion',
    service_tier=None,
    system_fingerprint=None,
    usage=CompletionUsage(completion_tokens=67, prompt_tokens=409, total_tokens=476, completion_tokens_details=None, prompt_tokens_details=None),
    prompt_logprobs=None,
    prompt_token_ids=None,
    kv_transfer_params=None
)


MESSAGE2=[
    {'role': 'system', 'content': '你是一名人工智能助手，你擅长使用工具回答用户问题。'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30."},
    {
        'content': None,
        'refusal': None,
        'role': 'assistant',
        'annotations': None,
        'audio': None,
        'function_call': None,
        'tool_calls': [
            {
                'id': 'chatcmpl-tool-5e2ec27d2c0443ff919b058f8d4f1f9f',
                'function': {'arguments': '{"location": "San Francisco, California, USA"}', 'name': 'get_current_temperature'},
                'type': 'function'
            },
            {
                'id': 'chatcmpl-tool-ce05bda54780485da364b820ae9f5a2f',
                'function': {'arguments': '{"location": "San Francisco, California, USA", "date": "2024-10-01"}', 'name': 'get_temperature_date'},
                'type': 'function'
            }
        ],
        'reasoning_content': None
    },
    {
        'role': 'tool',
        'content': '{"temperature": 26.1, "location": "San Francisco, California, USA", "unit": "celsius"}',
        'tool_call_id': 'chatcmpl-tool-5e2ec27d2c0443ff919b058f8d4f1f9f'
    },
    {
        'role': 'tool',
        'content': '{"temperature": 25.9, "location": "San Francisco, California, USA", "date": "2024-10-01", "unit": "celsius"}',
        'tool_call_id': 'chatcmpl-tool-ce05bda54780485da364b820ae9f5a2f'
    }
]

RESPONSE2=ChatCompletion(
    id='chatcmpl-76207bc38a0b47368f8543b6f0d4a6cf',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='The current temperature in San Francisco is 26.1°C. Tomorrow, the temperature is expected to be 25.9°C.',
                refusal=None,
                role='assistant',
                annotations=None,
                audio=None,
                function_call=None,
                tool_calls=[],
                reasoning_content=None
            ),
            stop_reason=None,
            token_ids=None
        )
    ],
    created=1770972259,
    model='Qwen3-30B-A3B-Instruct-2507-2',
    object='chat.completion',
    service_tier=None,
    system_fingerprint=None,
    usage=CompletionUsage(completion_tokens=30, prompt_tokens=561, total_tokens=591, completion_tokens_details=None, prompt_tokens_details=None),
    prompt_logprobs=None,
    prompt_token_ids=None,
    kv_transfer_params=None
)

rounds2_raw_prompt="""
<|im_start|>system
你是一名人工智能助手，你擅长使用工具回答用户问题。

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \"City, State, Country\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \"Year-Month-Day\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \"celsius\"."}}, "required": ["location", "date"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow? Current Date: 2024-09-30.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, California, USA"}}
</tool_call>
<tool_call>
{"name": "get_temperature_date", "arguments": {"location": "San Francisco, California, USA", "date": "2024-10-01"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 26.1, "location": "San Francisco, California, USA", "unit": "celsius"}
</tool_response>
<tool_response>
{"temperature": 25.9, "location": "San Francisco, California, USA", "date": "2024-10-01", "unit": "celsius"}
</tool_response><|im_end|>
<|im_start|>assistant
"""
