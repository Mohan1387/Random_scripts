#---------------------------------
"""
/fastapi-api-key
├── /app
│   ├── main.py  # Main FastAPI application
│   └── models.py  # Database models and connection
└── /venv (optional)
"""
# app/models.py
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import secrets

# Database connection
SQLALCHEMY_DATABASE_URL = "sqlite:///./api_keys.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Base model
Base = declarative_base()

# API Key Model
class APIKey(Base):
    __tablename__ = "api_keys"

    key = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    expires_at = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

# Session local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ------------------------
# Generate API Key Utility
# ------------------------
def generate_api_key(user_id: str, validity_days: int = 365):
    """Generate a secure API key valid for 1 year and save to DB."""
    db = SessionLocal()
    api_key = secrets.token_urlsafe(32)
    expiry_date = datetime.utcnow() + timedelta(days=validity_days)

    # Create and store API key in the database
    new_api_key = APIKey(key=api_key, user_id=user_id, expires_at=expiry_date)
    db.add(new_api_key)
    db.commit()
    db.close()

    return api_key, expiry_date

# ------------------------
# Validate API Key Utility
# ------------------------
def validate_api_key(api_key: str):
    """Validate API key and check expiration."""
    db = SessionLocal()
    api_key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    db.close()

    if not api_key_record:
        return None
    if datetime.utcnow() > api_key_record.expires_at:
        return "expired"
    return api_key_record


#----------------------------------------------------------------------------

# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Request
from app.models import generate_api_key, validate_api_key
from sqlalchemy.orm import Session

app = FastAPI()

# ------------------------
# Endpoint: Generate API Key
# ------------------------
@app.post("/generate-api-key/")
def create_api_key(user_id: str):
    """Generate and store an API key for the given user."""
    api_key, valid_until = generate_api_key(user_id)
    return {"api_key": api_key, "valid_until": valid_until}

# ------------------------
# Middleware: API Key Validation
# ------------------------
def validate_api_key_middleware(request: Request):
    """Middleware to validate API key from header."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_info = validate_api_key(api_key)
    if api_info is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    elif api_info == "expired":
        raise HTTPException(status_code=401, detail="API key expired")

    return api_info

# ------------------------
# Protected Endpoint
# ------------------------
@app.get("/protected-data/")
def get_protected_data(api_info: dict = Depends(validate_api_key_middleware)):
    """Return protected data if the API key is valid."""
    return {
        "message": "Access granted!",
        "user_id": api_info.user_id,
        "expires_at": api_info.expires_at,
    }

# ------------------------
# Run with: uvicorn app.main:app --reload
# ------------------------

#---------------------------------------
def revoke_api_key(api_key: str):
    """Revoke API key by deleting it from the DB."""
    db = SessionLocal()
    api_key_record = db.query(APIKey).filter(APIKey.key == api_key).first()
    if api_key_record:
        db.delete(api_key_record)
        db.commit()
    db.close()
  #-----------------------------------------


# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session
from app.models import generate_api_key, validate_api_key
from typing import Dict

app = FastAPI()

# ------------------------
# Admin Credentials
# ------------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "securepassword123"

# Security for Basic Authentication
security = HTTPBasic()

# ------------------------
# Admin Authentication Dependency
# ------------------------
def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate admin using Basic Auth."""
    if (
        credentials.username != ADMIN_USERNAME
        or credentials.password != ADMIN_PASSWORD
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# ------------------------
# Endpoint: Generate API Key (Admin Only)
# ------------------------
@app.post("/generate-api-key/")
def create_api_key(user_id: str, auth: bool = Depends(authenticate_admin)):
    """Generate and store an API key for the given user (admin only)."""
    api_key, valid_until = generate_api_key(user_id)
    return {"api_key": api_key, "valid_until": valid_until}

# ------------------------
# Middleware: API Key Validation
# ------------------------
def validate_api_key_middleware(request: Request):
    """Middleware to validate API key from header."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_info = validate_api_key(api_key)
    if api_info is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    elif api_info == "expired":
        raise HTTPException(status_code=401, detail="API key expired")

    return api_info

# ------------------------
# Protected Endpoint
# ------------------------
@app.get("/protected-data/")
def get_protected_data(api_info: Dict = Depends(validate_api_key_middleware)):
    """Return protected data if the API key is valid."""
    return {
        "message": "Access granted!",
        "user_id": api_info.user_id,
        "expires_at": api_info.expires_at,
    }

# ------------------------
# Run with: uvicorn app.main:app --reload
# ------------------------

