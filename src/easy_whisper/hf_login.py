import os
from typing import Optional
from getpass import getpass
from huggingface_hub import HfFolder, login, whoami
from easy_whisper import logger as main_logger

# Set up logging
logger = main_logger.getChild("hf_login")

def ensure_hf_login(force_login: bool = False, token: Optional[str] = None) -> bool:
    """
    Ensure the user is logged in to Hugging Face Hub.
    
    Args:
        force_login (bool): Force a new login even if a token already exists
        token (str, optional): Hugging Face token to use. If None, will check for
                               environment variable, cached token, or prompt user.
        
    Returns:
        bool: True if login was successful, False otherwise
    """
    try:
        # Check if token is provided directly
        if token is not None:
            logger.info("Using provided Hugging Face token")
            login(token=token, add_to_git_credential=True)
            try:
                # Verify login by checking user info
                user_info = whoami()
                logger.info(f"Successfully logged in as: {user_info.get('name', 'Unknown')}")
                return True
            except:
                logger.error("Failed to validate the provided token")
                return False
                
        # Check if already logged in via environment variable
        if "HUGGING_FACE_HUB_TOKEN" in os.environ and not force_login:
            token = os.environ["HUGGING_FACE_HUB_TOKEN"]
            logger.info("Using Hugging Face token from environment variable")
            login(token=token, add_to_git_credential=True)
            return True
            
        # Check if token exists in cache and is valid
        token = HfFolder.get_token()
        if token is not None and not force_login:
            try:
                # Verify token is valid with a simple API call
                user_info = whoami()
                logger.info(f"Already logged in to Hugging Face Hub as: {user_info.get('name', 'Unknown')}")
                return True
            except:
                logger.warning("Cached token exists but appears to be invalid")
                # Continue to token prompt
        
        # Token doesn't exist, is invalid, or force_login is True
        logger.info("Please log in to Hugging Face Hub to push models")
        print("\n=== Hugging Face Hub Login ===")
        print("You can find your token at https://huggingface.co/settings/tokens")
        token = getpass("Enter your Hugging Face token: ")
        
        if token.strip() == "":
            logger.warning("Empty token provided, skipping login")
            return False
        
        # Attempt to log in with the provided token
        login(token=token, add_to_git_credential=True)
        
        # Verify login was successful
        try:
            user_info = whoami()
            logger.info(f"Successfully logged in as: {user_info.get('name', 'Unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to validate login: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error during Hugging Face login: {e}")
        return False

def get_hf_username() -> Optional[str]:
    """
    Get the username of the currently logged in Hugging Face user.
    
    Returns:
        Optional[str]: Username if logged in, None otherwise
    """
    try:
        user_info = whoami()
        return user_info.get('name')
    except:
        return None

def is_logged_in() -> bool:
    """
    Check if user is currently logged in to Hugging Face Hub.
    
    Returns:
        bool: True if logged in, False otherwise
    """
    try:
        whoami()
        return True
    except:
        return False

def logout_hf():
    """
    Log out from Hugging Face Hub by removing token.
    """
    try:
        # Clear token from HfFolder
        token_path = HfFolder.path_token
        if os.path.exists(token_path):
            os.remove(token_path)
            logger.info("Logged out from Hugging Face Hub")
        else:
            logger.info("No login token found")
    except Exception as e:
        logger.error(f"Error during logout: {e}")