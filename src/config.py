import os


PROJECT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


SERP_API_KEY = os.getenv('SERP_API_KEY')
