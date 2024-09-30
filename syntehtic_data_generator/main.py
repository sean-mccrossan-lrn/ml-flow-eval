from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class StudentQuestionsDEI(BaseModel):
    question: str
    bias_label: bool
    bias_score: float
    bias_axes: list[str]
    bias_explanation: str
    comment: str
    question_type: str
    debias_question: str