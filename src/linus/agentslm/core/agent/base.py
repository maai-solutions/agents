from typing import Union, TypeVar, Type
from pydantic import BaseModel

from openai import AsyncOpenAI, OpenAI

from agentslm.core.interfaces.logger import ILogger

IT = TypeVar('IT')
OT = TypeVar('OT')

class AgentBase[IT, OT]:
    def __init__(
            self,
            name: str,
            llm: Union[AsyncOpenAI, OpenAI],
            model: str,
            input_type: Type[IT] = None,
            output_type: Type[OT] = None,
            instructions: str | None = None,
            logger: ILogger | None = None
        ):
            self.name = name
            if name is None or name.strip() == "":
                raise ValueError("Agent name must be provided and cannot be empty.")
            
            self.llm = llm
            if llm is None:
                raise ValueError("An LLM instance must be provided.")
            
            self.model = model
            if model is None or model.strip() == "":
                raise ValueError("Model name must be provided and cannot be empty.")
            
            self.input_type = input_type or str
            self.output_type = output_type or str
            self.instructions = instructions or "You are a helpful assistant."

            if logger is None:
                 raise ValueError("A logger instance must be provided.")
            
            self.log = logger 

    def act(self, input_data: IT) -> OT:
        """
        Process input data through the LLM and return structured output.

        Args:
            input_data: Input data conforming to the IT type (Pydantic model or str)

        Returns:
            Structured output conforming to the OT type (Pydantic model or str)
        """
        # Handle input conversion
        if isinstance(input_data, str):
            input_json = input_data
        elif hasattr(input_data, 'model_dump_json'):
            # Pydantic model
            input_json = input_data.model_dump_json(indent=2)
        else:
            # Fallback for other types
            input_json = str(input_data)

        # Create the prompt with structured input
        messages = [
            {
                "role": "system",
                "content": self.instructions or "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Context:\n{input_json}\n\nProcess this input and provide a structured response."
            }
        ]

        # Handle output type - check if it's str type
        if self.output_type is str or self.output_type == str:
            # For string output, use regular completion
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        else:
            # For Pydantic model output, use structured output
            response = self.llm.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=self.output_type
            )
            return response.choices[0].message.parsed
        