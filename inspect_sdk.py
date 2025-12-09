
import os
from google import genai
from google.genai import types

def inspect_sdk():
    print("Testing ImportFileConfig with stringListValue...")
    try:
        config = types.ImportFileConfig(
            custom_metadata=[
                {"key": "cat", "string_list_value": ["a", "b"]},
                {"key": "url", "string_value": "http://example.com"}
            ]
        )
        print("Success! Config created:")
        print(config)
    except Exception as e:
        print(f"Error creating config: {e}")

if __name__ == "__main__":
    inspect_sdk()
