from utils.config import GenerativeAiModel, AiCloudHost, AiProvider, AiModel
from utils.aws_bedrock import call_bedrock_model_with_langchain
from utils.openai import call_openai_model

def model_factory(model_config, prompt, data):
    """
    Model factory to call either AWS Bedrock or OpenAI models.
    """
    if model_config.cloud_host == AiCloudHost.AWS:
        return call_bedrock_model_with_langchain(model_config, prompt, data)
    elif model_config.cloud_host == AiCloudHost.OpenAI:
        return call_openai_model(model_config, prompt, data)
    else:
        raise ValueError("Unsupported cloud host.")

if __name__ == "__main__":
    aws_model = GenerativeAiModel(
        cloud_host=AiCloudHost.AWS, 
        provider=AiProvider.Anthropic, 
        model=AiModel.CLAUDE_35, 
        temperature=0.7
    )

    openai_model = GenerativeAiModel(
        cloud_host=AiCloudHost.OpenAI,
        provider=AiProvider.OpenAI,
        model=AiModel.GPT4, 
        temperature=0.7
    )

    print(f"Using model: {openai_model.actual_model_name}")

    prompt = "Translate from English to Spanish"
    data = "Hi, how are you?"
  
    response = model_factory(openai_model, prompt, data)
    
    print("Model Response:")
    print(response)
