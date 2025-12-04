"""
Unit tests for Kalshi RSA authentication.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshiflow.auth import RSASigner, KalshiAuth, KalshiAuthError


class TestRSASigner:
    """Test RSASigner functionality."""
    
    @pytest.fixture
    def temp_key_file(self):
        """Create a temporary RSA private key file for testing."""
        # Generate a test RSA key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Serialize the key
        pem_data = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem') as f:
            f.write(pem_data)
            temp_file_path = f.name
        
        yield temp_file_path
        
        # Cleanup
        os.unlink(temp_file_path)
    
    def test_load_private_key_success(self, temp_key_file):
        """Test successful private key loading."""
        signer = RSASigner(temp_key_file)
        assert signer._private_key is not None
        assert isinstance(signer._private_key, rsa.RSAPrivateKey)
    
    def test_load_private_key_file_not_found(self):
        """Test error when private key file doesn't exist."""
        with pytest.raises(KalshiAuthError, match="Private key file not found"):
            RSASigner("/nonexistent/path/to/key.pem")
    
    def test_load_invalid_private_key(self):
        """Test error when private key file is invalid."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as f:
            f.write("invalid key content")
            temp_file_path = f.name
        
        try:
            with pytest.raises(KalshiAuthError, match="Failed to load private key"):
                RSASigner(temp_file_path)
        finally:
            os.unlink(temp_file_path)
    
    def test_sign_message_success(self, temp_key_file):
        """Test successful message signing."""
        signer = RSASigner(temp_key_file)
        message = "test message"
        
        signature = signer.sign(message)
        
        assert isinstance(signature, str)
        assert len(signature) > 0
        # Base64 encoded signature should only contain valid base64 characters
        import base64
        try:
            base64.b64decode(signature)
        except Exception:
            pytest.fail("Signature is not valid base64")
    
    def test_sign_empty_message(self, temp_key_file):
        """Test signing empty message."""
        signer = RSASigner(temp_key_file)
        signature = signer.sign("")
        
        assert isinstance(signature, str)
        assert len(signature) > 0
    
    def test_sign_without_loaded_key(self, temp_key_file):
        """Test signing when private key is not loaded."""
        signer = RSASigner(temp_key_file)
        signer._private_key = None  # Simulate key not loaded
        
        with pytest.raises(KalshiAuthError, match="Private key not loaded"):
            signer.sign("test")


class TestKalshiAuth:
    """Test KalshiAuth functionality."""
    
    @pytest.fixture
    def temp_key_file(self):
        """Create a temporary RSA private key file for testing."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        pem_data = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pem') as f:
            f.write(pem_data)
            temp_file_path = f.name
        
        yield temp_file_path
        os.unlink(temp_file_path)
    
    def test_init_success(self, temp_key_file):
        """Test successful KalshiAuth initialization."""
        auth = KalshiAuth("test_api_key", temp_key_file)
        
        assert auth.api_key_id == "test_api_key"
        assert isinstance(auth.signer, RSASigner)
    
    def test_from_env_success(self, temp_key_file):
        """Test creating KalshiAuth from environment variables."""
        with patch.dict(os.environ, {
            'KALSHI_API_KEY_ID': 'test_api_key',
            'KALSHI_PRIVATE_KEY_PATH': temp_key_file
        }):
            auth = KalshiAuth.from_env()
            
            assert auth.api_key_id == "test_api_key"
            assert auth.signer.private_key_path == Path(temp_key_file)
    
    def test_from_env_missing_api_key(self, temp_key_file):
        """Test error when KALSHI_API_KEY_ID is missing."""
        with patch.dict(os.environ, {
            'KALSHI_PRIVATE_KEY_PATH': temp_key_file
        }, clear=True):
            with pytest.raises(KalshiAuthError, match="KALSHI_API_KEY_ID environment variable is required"):
                KalshiAuth.from_env()
    
    def test_from_env_missing_private_key_path(self):
        """Test error when KALSHI_PRIVATE_KEY_PATH is missing."""
        with patch.dict(os.environ, {
            'KALSHI_API_KEY_ID': 'test_api_key'
        }, clear=True):
            with pytest.raises(KalshiAuthError, match="KALSHI_PRIVATE_KEY_PATH environment variable is required"):
                KalshiAuth.from_env()
    
    def test_create_auth_headers(self, temp_key_file):
        """Test authentication header creation."""
        auth = KalshiAuth("test_api_key", temp_key_file)
        
        # Mock the signer to return a predictable signature
        with patch.object(auth.signer, 'sign', return_value='mock_signature'):
            headers = auth.create_auth_headers("GET", "/test/path")
        
        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        
        assert headers["KALSHI-ACCESS-KEY"] == "test_api_key"
        assert headers["KALSHI-ACCESS-SIGNATURE"] == "mock_signature"
        
        # Timestamp should be numeric
        timestamp = headers["KALSHI-ACCESS-TIMESTAMP"]
        assert timestamp.isdigit()
        assert len(timestamp) == 13  # Milliseconds timestamp
    
    @patch('time.time', return_value=1701648000.123)
    def test_signature_payload_format(self, mock_time, temp_key_file):
        """Test that signature payload is formatted correctly."""
        auth = KalshiAuth("test_api_key", temp_key_file)
        
        # Capture the actual signature payload
        captured_payload = None
        
        def mock_sign(payload):
            nonlocal captured_payload
            captured_payload = payload
            return "mock_signature"
        
        with patch.object(auth.signer, 'sign', side_effect=mock_sign):
            auth.create_auth_headers("GET", "/test/path")
        
        # Verify signature payload format: timestamp_ms + method + path
        expected_payload = "1701648000123GET/test/path"
        assert captured_payload == expected_payload
    
    def test_create_websocket_auth_message(self, temp_key_file):
        """Test WebSocket authentication message creation."""
        auth = KalshiAuth("test_api_key", temp_key_file)
        
        with patch.object(auth.signer, 'sign', return_value='mock_signature'):
            ws_auth = auth.create_websocket_auth_message()
        
        # Should use same format as regular auth headers but with "/" path
        assert "KALSHI-ACCESS-KEY" in ws_auth
        assert "KALSHI-ACCESS-SIGNATURE" in ws_auth
        assert "KALSHI-ACCESS-TIMESTAMP" in ws_auth
        
        assert ws_auth["KALSHI-ACCESS-KEY"] == "test_api_key"
    
    def test_multiple_auth_headers_different_timestamps(self, temp_key_file):
        """Test that multiple calls generate different timestamps."""
        auth = KalshiAuth("test_api_key", temp_key_file)
        
        with patch.object(auth.signer, 'sign', return_value='mock_signature'):
            headers1 = auth.create_auth_headers("GET", "/path1")
            # Small delay to ensure different timestamp
            import time
            time.sleep(0.001)
            headers2 = auth.create_auth_headers("GET", "/path2")
        
        # Timestamps should be different
        assert headers1["KALSHI-ACCESS-TIMESTAMP"] != headers2["KALSHI-ACCESS-TIMESTAMP"]


class TestAuthErrorHandling:
    """Test error handling in authentication modules."""
    
    def test_kalshi_auth_error_inheritance(self):
        """Test that KalshiAuthError is a proper Exception."""
        error = KalshiAuthError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"
    
    def test_error_propagation_from_signer(self):
        """Test that errors from RSASigner are properly propagated."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pem') as f:
            f.write("invalid content")
            temp_file_path = f.name
        
        try:
            with pytest.raises(KalshiAuthError):
                KalshiAuth("test_key", temp_file_path)
        finally:
            os.unlink(temp_file_path)