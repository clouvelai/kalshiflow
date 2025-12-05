#!/usr/bin/env python3
"""
Script to add enhanced debug logging to help diagnose the Render connection issues.
This will temporarily modify the database.py file to add more detailed logging.
"""

import os

def enhance_database_logging():
    """Add enhanced debug logging to database.py"""
    
    backend_db_path = "/Users/samuelclark/Desktop/kalshiflow/backend/src/kalshiflow/database.py"
    
    # Read the current file
    with open(backend_db_path, 'r') as f:
        content = f.read()
    
    # Find the initialize method and add debug logging
    old_logging = '''        # Log which database URL is being used for debugging
        pooled_url = os.getenv("DATABASE_URL_POOLED")
        direct_url = os.getenv("DATABASE_URL")
        logger.info(f"Database URL priority: pooled={'***' if pooled_url else 'None'}, direct={'***' if direct_url else 'None'}")
        logger.info(f"Using database URL: {self.database_url[:50]}... (host: {self.database_url.split('@')[1].split(':')[0] if '@' in self.database_url else 'unknown'})")'''
    
    new_logging = '''        # Enhanced debugging for Render connectivity issues
        pooled_url = os.getenv("DATABASE_URL_POOLED")
        direct_url = os.getenv("DATABASE_URL")
        
        logger.info("=== DATABASE CONNECTION DEBUG ===")
        logger.info(f"Environment: ENVIRONMENT={os.getenv('ENVIRONMENT', 'unknown')}")
        logger.info(f"DATABASE_URL_POOLED available: {'YES' if pooled_url else 'NO'}")
        logger.info(f"DATABASE_URL available: {'YES' if direct_url else 'NO'}")
        
        if pooled_url:
            logger.info(f"POOLED URL: {pooled_url}")
        if direct_url:
            logger.info(f"DIRECT URL: {direct_url}")
            
        logger.info(f"SELECTED URL: {self.database_url}")
        
        # Parse URL details
        if self.database_url:
            try:
                url_parts = self.database_url.split('@')
                if len(url_parts) > 1:
                    host_port = url_parts[1].split('/')[0]
                    host = host_port.split(':')[0]
                    port = host_port.split(':')[1] if ':' in host_port else 'unknown'
                    logger.info(f"TARGET HOST: {host}")
                    logger.info(f"TARGET PORT: {port}")
                    
                    # Check if it's the pooled or direct connection
                    if ':6543' in self.database_url:
                        logger.info("CONNECTION TYPE: POOLED (6543) - Should work with IPv4")
                    elif ':5432' in self.database_url:
                        logger.info("CONNECTION TYPE: DIRECT (5432) - May have IPv6 issues")
                    else:
                        logger.info(f"CONNECTION TYPE: UNKNOWN PORT ({port})")
                else:
                    logger.info("URL FORMAT: Could not parse host/port")
            except Exception as parse_error:
                logger.info(f"URL PARSING ERROR: {parse_error}")
        
        logger.info("=== ATTEMPTING CONNECTION ===")'''
    
    # Replace the logging section
    if old_logging in content:
        content = content.replace(old_logging, new_logging)
        
        # Also add a post-connection log
        connection_attempt = '''        try:
            # Create connection pool
            # Set statement_cache_size=0 to work with Supabase pgbouncer pooling
            self._pool = await asyncpg.create_pool('''
        
        enhanced_connection_attempt = '''        try:
            logger.info("Creating asyncpg connection pool...")
            # Create connection pool
            # Set statement_cache_size=0 to work with Supabase pgbouncer pooling
            self._pool = await asyncpg.create_pool('''
        
        content = content.replace(connection_attempt, enhanced_connection_attempt)
        
        # Add error logging enhancement
        error_logging = '''        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise'''
        
        enhanced_error_logging = '''        except Exception as e:
            logger.error("=== DATABASE CONNECTION FAILED ===")
            logger.error(f"ERROR TYPE: {type(e).__name__}")
            logger.error(f"ERROR MESSAGE: {str(e)}")
            logger.error(f"ATTEMPTED URL: {self.database_url}")
            
            # Additional error context
            if "Network is unreachable" in str(e):
                logger.error("DIAGNOSIS: Network connectivity issue - likely IPv4/IPv6 or firewall problem")
            elif "Connection refused" in str(e):
                logger.error("DIAGNOSIS: Service is not running or port is blocked")
            elif "timeout" in str(e).lower():
                logger.error("DIAGNOSIS: Connection timeout - slow network or overloaded service")
            elif "authentication" in str(e).lower():
                logger.error("DIAGNOSIS: Authentication failed - check credentials")
            else:
                logger.error("DIAGNOSIS: Unknown connection error")
                
            logger.error("=== END DATABASE DEBUG ===")
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise'''
        
        content = content.replace(error_logging, enhanced_error_logging)
        
        # Write the enhanced file
        with open(backend_db_path, 'w') as f:
            f.write(content)
        
        print("✅ Enhanced debug logging added to database.py")
        print("This will provide detailed connection diagnostics on Render")
        return True
    else:
        print("❌ Could not find the logging section to enhance")
        return False

if __name__ == "__main__":
    enhance_database_logging()