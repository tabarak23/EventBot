# tests/conftest.py
import pytest
from unittest.mock import MagicMock

from src.backend.config import Config as RealConfig


@pytest.fixture
def mock_chatbot_agent_instance(mocker):
    mocker.patch('pinecone.Pinecone')
    mocker.patch('google.generativeai.configure')
    mocker.patch('google.generativeai.GenerativeModel')
    mocker.patch('langchain_google_genai.GoogleGenerativeAIEmbeddings')
    mocker.patch('langchain_pinecone.PineconeVectorStore')
    mocker.patch('langchain_community.document_loaders.PyPDFLoader')
    mocker.patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    mocker.patch('langchain.prompts.ChatPromptTemplate')

    mock_config_instance = MagicMock(spec=RealConfig)
    mock_config_instance.PINECONE_API_KEY = 'test_pinecone_key'
    mock_config_instance.GEMINI_API_KEY = 'test_gemini_key'
    mock_config_instance.PINECONE_INDEX_NAME = 'test-index'
    
    mock_config_instance.ALLOWED_EXTENSIONS = ["pdf"]
    mock_config_instance.MAX_FILE_SIZE = 2 * 1024 * 1024
    mock_config_instance.HOST = "0.0.0.0"
    mock_config_instance.PORT = 5000
    mock_config_instance.DEBUG = False
    mock_config_instance.ENDPOINT = "http://localhost:5000"
    mock_config_instance.DATABASE_USER = "test_user"
    mock_config_instance.DATABASE_PASSWORD = "test_password"
    mock_config_instance.DATABASE_HOST = "test_host"
    mock_config_instance.DATABASE_PORT = "3306"
    mock_config_instance.DATABASE_NAME = "test_db"
    mock_config_instance.PINECONE_CLOUD = "aws"
    mock_config_instance.PINECONE_REGION = "us-east-1"
    mock_config_instance.PINECONE_DIMENSION = 768

    mock_config_instance.validate_pinecone_config.return_value = None
    mock_config_instance.validate_gemini_config.return_value = None
    mock_config_instance.validate_database_config.return_value = None
    mock_config_instance.validate_required_env_vars.return_value = None

    mock_config_instance.database_url = "mysql+pymysql://test_user:test_password@test_host:3306/test_db?charset=utf8mb4"

    mocker.patch('src.backend.config.Config', return_value=mock_config_instance)
    
    agent = ChatbotAgent()
    return agent

@pytest.fixture(scope="function")
def app_client(mocker):
    from src.backend.main import app as main_app
    from fastapi.testclient import TestClient

    mock_config_instance = MagicMock(spec=RealConfig)
    mock_config_instance.PINECONE_API_KEY = 'test_pinecone_key'
    mock_config_instance.GEMINI_API_KEY = 'test_gemini_key'
    mock_config_instance.PINECONE_INDEX_NAME = 'test-index'
    mock_config_instance.validate_pinecone_config.return_value = None
    mock_config_instance.validate_gemini_config.return_value = None
    mock_config_instance.validate_database_config.return_value = None
    mock_config_instance.validate_required_env_vars.return_value = None
    mock_config_instance.database_url = "mysql+pymysql://test_user:test_password@test_host:3306/test_db?charset=utf8mb4"
    
    mocker.patch('src.backend.config.Config', return_value=mock_config_instance)

    with TestClient(main_app) as client:
        yield client