# tests/conftest.py

import pytest
import os
# Ensure ChatbotAgent is imported here, as it's used in the fixture below.
from src.backend.agents.rag_agent import ChatbotAgent 
from unittest.mock import MagicMock

# Import necessary components from your application
from src.backend import create_app # Assuming this is correct for your Flask/FastAPI app

# Import Config to use as a spec for the mock
from src.backend.config import Config as RealConfig


@pytest.fixture(scope='session', autouse=True)
def setup_test_env_vars():
    """
    Fixture to ensure environment variables for testing are set.
    This helps prevent the actual ChatbotAgent from trying to connect
    to real external services during app client tests where it's mocked.
    For agent tests, we'll mock individual components.
    """
    os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
    os.environ['PINECONE_API_KEY'] = 'test_pinecone_key'
    os.environ['PINECONE_INDEX'] = 'test-index' # Changed to PINECONE_INDEX as per your config.py (not PINECONE_INDEX_NAME for env var)
    os.environ['FLASK_ENV'] = 'testing'
    
    # Yield control to tests
    yield
    
    # Clean up environment variables after tests
    del os.environ['GEMINI_API_KEY']
    del os.environ['PINECONE_API_KEY']
    del os.environ['PINECONE_INDEX']
    del os.environ['FLASK_ENV']

@pytest.fixture(scope='function')
def app_client(mocker):
    """
    Fixture for a Flask test client with a mocked ChatbotAgent.
    
    This fixture ensures that when testing Flask routes, the actual
    ChatbotAgent's external dependencies (Gemini, Pinecone) are NOT
    called. Instead, we control its behavior through mocks.
    """
    # Create a mock for the ChatbotAgent instance
    mock_chatbot_agent = mocker.Mock(spec=ChatbotAgent)
    
    # Ensure __init__ doesn't try to connect to real services
    # We mock out the internal initialization methods.
    mock_chatbot_agent._initialize_pinecone = MagicMock()
    mock_chatbot_agent._initialize_gemini = MagicMock()
    mock_chatbot_agent._initialize_embeddings = MagicMock()
    mock_chatbot_agent._validate_credentials = MagicMock() # Mock validation
    mock_chatbot_agent._setup_prompt_template = MagicMock() # Mock prompt setup

    # When create_app is called, we want it to use our mocked agent.
    # We will patch the ChatbotAgent class during app creation.
    mocker.patch('src.backend.agents.rag_agent.ChatbotAgent', return_value=mock_chatbot_agent)
    
    # Patch the initialization of ChatbotAgent in __init__.py directly
    # This ensures that `app.chatbot_agent` gets our mock.
    # Adjust path if your app initialization is different.
    mocker.patch('src.backend.__init__.ChatbotAgent', return_value=mock_chatbot_agent)

    app = create_app()
    app.config['TESTING'] = True
    
    # Explicitly assign the mock to the app context if needed (though patching should handle it)
    app.chatbot_agent = mock_chatbot_agent 

    with app.test_client() as client:
        yield client, mock_chatbot_agent

@pytest.fixture
def mock_chatbot_agent_instance(mocker):
    """
    Fixture to provide a mocked ChatbotAgent instance for direct agent testing.
    This fixture is specifically for tests *of the ChatbotAgent itself*,
    where you want to mock its internal dependencies (Pinecone, Gemini, Langchain components)
    rather than mocking the agent itself.
    """
    # Mock external library components that ChatbotAgent depends on
    mocker.patch('pinecone.Pinecone')
    mocker.patch('google.generativeai.configure')
    mocker.patch('google.generativeai.GenerativeModel')
    mocker.patch('langchain_google_genai.GoogleGenerativeAIEmbeddings')
    mocker.patch('langchain_pinecone.PineconeVectorStore')
    mocker.patch('langchain_community.document_loaders.PyPDFLoader')
    mocker.patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    mocker.patch('langchain.prompts.ChatPromptTemplate')

    # --- CRITICAL CHANGE HERE: Mock the Config class itself ---
    # Instead of relying on os.environ, we directly control the Config object
    mock_config_instance = MagicMock(spec=RealConfig)
    mock_config_instance.PINECONE_API_KEY = 'test_pinecone_key'
    mock_config_instance.GEMINI_API_KEY = 'test_gemini_key'
    # Use PINECONE_INDEX_NAME as this is what Config class reads from its attribute
    mock_config_instance.PINECONE_INDEX_NAME = 'test-index' 
    
    # Set other necessary config attributes for the mock
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

    # Mock validation methods to prevent them from raising errors
    mock_config_instance.validate_pinecone_config.return_value = None
    mock_config_instance.validate_gemini_config.return_value = None
    mock_config_instance.validate_database_config.return_value = None
    mock_config_instance.validate_required_env_vars.return_value = None

    # Mock any properties like database_url
    mock_config_instance.database_url = "mysql+pymysql://test_user:test_password@test_host:3306/test_db?charset=utf8mb4"

    # Patch the Config class where ChatbotAgent imports it from.
    # Assuming ChatbotAgent does `from ..config import Config`
    mocker.patch('src.backend.config.Config', return_value=mock_config_instance)
    
    # --- Remove these lines as they are no longer necessary and can cause issues ---
    # os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
    # os.environ['PINECONE_API_KEY'] = 'test_pinecone_key'
    # os.environ['PINECONE_INDEX'] = 'test-index'
    
    agent = ChatbotAgent() # This will now get the mocked Config instance
    
    # --- Remove these cleanup lines as Config is mocked, not relying on os.environ directly ---
    # del os.environ['GEMINI_API_KEY']
    # del os.environ['PINECONE_API_KEY']
    # del os.environ['PINECONE_INDEX']

    yield agent