from enum import Enum
from pydantic import BaseModel, Field

class AiCloudHost(str, Enum):
    Azure = "Azure"
    AWS = "AWS"
    OpenAI = "OpenAI"
    OLLama = "OLLama"  # Local LLM provider


class AiProvider(str, Enum):
    AzureAI = "AzureAI"
    OpenAI = "OpenAI"
    HuggingFace = "HuggingFace"
    Anthropic = "Anthropic"
    Meta = "Meta"
    Google = "Google"


class AiModelType(str, Enum):
    """
    Enum for the type of AI model
    """

    EMBEDDING = "EMBEDDING"  # Embedding model takes text input and generates a fixed-size vector output
    GPT = "GPT"  # GPT model takes text input and generates text output
    GPT_MULTIMODAL = "GPT_MULTIMODAL"  # GPT model with multimodal capabilities takes both text and image inputs
    IMAGE = "IMAGE"  # Convert Text to Image
    VIDEO = "VIDEO"  # Convert Text to Video
    AUDIO = "AUDIO"  # Convert Text to Audio
    TTS = "TTS"  # Text to Spoken Audio
    STT = "STT"  # Speech to Text


class ContextWindow(BaseModel):
    """
    Context window for the AI model represented in tokens
    """

    input_tokens: int = Field(0, description="Input tokens")
    output_tokens: int = Field(0, description="Output tokens")

class AiModel(str, Enum):
    CLAUDE_2 = ("Claude-2", 100_000, 4_096, AiModelType.GPT)
    CLAUDE_3 = ("Claude-3", 200_000, 4_096, AiModelType.GPT_MULTIMODAL)
    CLAUDE_35 = ("Claude-35", 200_000, 4_096, AiModelType.GPT_MULTIMODAL)
    GPT3Turbo = ("GPT3.5-Turbo", 16_385, 4_096, AiModelType.GPT)
    GPT4 = ("GPT4", 8_192, 4_096, AiModelType.GPT_MULTIMODAL)
    GPT4Turbo = ("GPT4-Turbo", 128_000, 4_096, AiModelType.GPT_MULTIMODAL)
    GPT4o = ("GPT4o", 450_000, 4_096, AiModelType.GPT_MULTIMODAL)
    LLAMA_2 = ("LLAMA2", 4_096, 4_096, AiModelType.GPT)
    LLAMA_2_CHAT = ("LLAMA2-Chat", 4_096, 4_096, AiModelType.GPT)
    GEMMA = ("GEMMA", 8_192, 4_096, AiModelType.GPT)

    def __new__(cls, value, input_tokens, output_tokens, type=AiModelType.GPT):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.input_tokens = input_tokens
        obj.output_tokens = output_tokens
        obj.type = AiModelType(type)
        return obj

    @property
    def context_window(self) -> ContextWindow:
        return ContextWindow(input_tokens=self.input_tokens, output_tokens=self.output_tokens)

    @property
    def model_type(self) -> AiModelType:
        return self.type


# Step 2: Create a mapping from AiCloudHost and AiProvider to the actual model name.
CLOUD_PROVIDER_MODEL_MAPPING = {
    AiCloudHost.Azure: {
        AiProvider.AzureAI: {
            AiModel.GPT3Turbo: "GPT-35-Turbo",
            AiModel.GPT4: "GPT4",
        }
    },
    AiCloudHost.OpenAI: {
        AiProvider.OpenAI: {
            AiModel.GPT3Turbo: "gpt-3.5-turbo",
            AiModel.GPT4: "gpt-4",
            AiModel.GPT4Turbo: "gpt-4-turbo",
            AiModel.GPT4o: "gpt-4o",
        }
    },
    AiCloudHost.OLLama: {
        AiProvider.Meta: {
            AiModel.LLAMA_2: "llama2",
            AiModel.LLAMA_2_CHAT: "llama2:7b-chat",
        },
        AiProvider.Google: {AiModel.GEMMA: "gemma"},
    },
    AiCloudHost.AWS: {
        AiProvider.Anthropic: {
            AiModel.CLAUDE_35: "anthropic.claude-3-5-sonnet-20240620-v1:0",
            AiModel.CLAUDE_3: "anthropic.claude-3-sonnet-20240620-v1:0",
        }
    },
}


class GenerativeAiModel(BaseModel):
    cloud_host: AiCloudHost = AiCloudHost.Azure
    provider: AiProvider = AiProvider.AzureAI
    model: AiModel = AiModel.GPT3Turbo
    version: str = "1.0.0"
    temperature: float = 0.0

    @property
    def actual_model_name(self):
        return CLOUD_PROVIDER_MODEL_MAPPING[self.cloud_host][self.provider][self.model]

    @property
    def context_window(self) -> ContextWindow:
        return self.model.context_window

    @property
    def model_type(self) -> AiModelType:
        return self.model.model_type

    def __init__(self, **data):
        super().__init__(**data)
        if self.provider not in CLOUD_PROVIDER_MODEL_MAPPING[self.cloud_host]:
            raise ValueError(
                f"Provider '{self.provider}' is not available on '{self.cloud_host}'"
            )
        if (
            self.model
            not in CLOUD_PROVIDER_MODEL_MAPPING[self.cloud_host][self.provider]
        ):
            raise ValueError(
                f"Invalid model '{self.model}' for provider '{self.provider}' on '{self.cloud_host}'"
            )