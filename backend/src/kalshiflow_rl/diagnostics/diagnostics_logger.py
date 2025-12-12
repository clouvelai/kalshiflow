"""
M10 Consolidated Diagnostics Logging System.

Provides unified logging to both console (for SB3 integration) and file
for comprehensive training diagnostics and debugging.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, TextIO
from dataclasses import dataclass


@dataclass 
class DiagnosticsConfig:
    """Configuration for diagnostics logging."""
    session_id: Optional[int] = None
    algorithm: str = "unknown"
    log_level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    json_output: bool = True
    flush_frequency: int = 10  # Flush every N entries


class DiagnosticsLogger:
    """
    Consolidated logging system for M10 diagnostics.
    
    Features:
    - Dual output: console (for SB3) + file (for analysis)
    - Structured JSON logging for easy parsing
    - Organized artifact storage in session subdirectories
    - Real-time diagnostic summaries
    - Memory-efficient buffering
    """
    
    def __init__(
        self,
        output_dir: str,
        config: Optional[DiagnosticsConfig] = None,
        create_subdirs: bool = True
    ):
        """
        Initialize consolidated diagnostics logger.
        
        Args:
            output_dir: Base output directory for all diagnostics
            config: Logging configuration
            create_subdirs: Create organized subdirectory structure
        """
        self.config = config or DiagnosticsConfig()
        self.output_dir = Path(output_dir)
        
        # Create timestamped subdirectory for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_prefix = f"session{self.config.session_id}_" if self.config.session_id else ""
        
        if create_subdirs:
            self.run_dir = self.output_dir / f"{session_prefix}{self.config.algorithm}_{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = self.output_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging files
        self.diagnostics_log_path = self.run_dir / "diagnostics.log"
        self.json_log_path = self.run_dir / "diagnostics.jsonl"
        self.summary_path = self.run_dir / "training_summary.json"
        
        # Setup Python logger for console output
        self.logger = self._setup_console_logger()
        
        # File handles
        self.log_file: Optional[TextIO] = None
        self.json_file: Optional[TextIO] = None
        
        # Buffering for efficiency
        self.log_buffer = []
        self.buffer_count = 0
        
        # Training metadata
        self.training_metadata = {
            'session_id': self.config.session_id,
            'algorithm': self.config.algorithm,
            'start_time': datetime.now().isoformat(),
            'output_directory': str(self.run_dir),
            'diagnostics_files': {
                'text_log': str(self.diagnostics_log_path),
                'json_log': str(self.json_log_path),
                'summary': str(self.summary_path)
            }
        }
        
        # Initialize logging
        self._initialize_logging()
    
    def log_diagnostic(
        self,
        category: str,
        event_type: str, 
        data: Dict[str, Any],
        level: str = "INFO",
        console_summary: Optional[str] = None
    ) -> None:
        """
        Log a diagnostic event to both console and file.
        
        Args:
            category: Diagnostic category (action, reward, observation, episode, etc.)
            event_type: Type of event (step, episode_end, validation, etc.)
            data: Structured diagnostic data
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            console_summary: Optional concise summary for console output
        """
        timestamp = datetime.now()
        
        # Create structured log entry
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'category': category,
            'event_type': event_type,
            'level': level,
            'data': data
        }
        
        # Console logging (concise for SB3 integration)
        if self.config.console_output and console_summary:
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(f"[{category.upper()}] {console_summary}")
        
        # File logging (detailed)
        if self.config.file_output:
            self._write_to_log_file(timestamp, category, event_type, data, level)
        
        # JSON logging (structured)
        if self.config.json_output:
            self._write_to_json_file(log_entry)
        
        # Buffer management
        self.buffer_count += 1
        if self.buffer_count >= self.config.flush_frequency:
            self.flush()
    
    def log_action_event(self, action_data: Dict[str, Any], console_summary: str = None) -> None:
        """Log action-related diagnostic event."""
        self.log_diagnostic(
            category="action",
            event_type="action_step", 
            data=action_data,
            console_summary=console_summary or f"Action: {action_data.get('action_name', 'unknown')}"
        )
    
    def log_reward_event(self, reward_data: Dict[str, Any], console_summary: str = None) -> None:
        """Log reward-related diagnostic event."""
        self.log_diagnostic(
            category="reward",
            event_type="reward_step",
            data=reward_data,
            console_summary=console_summary or f"Reward: {reward_data.get('total_reward', 0.0):.2f}"
        )
    
    def log_observation_event(self, obs_data: Dict[str, Any], console_summary: str = None) -> None:
        """Log observation validation event."""
        self.log_diagnostic(
            category="observation",
            event_type="observation_validation",
            data=obs_data,
            level="WARNING" if not obs_data.get('observation_valid', True) else "DEBUG",
            console_summary=console_summary
        )
    
    def log_episode_summary(self, episode_data: Dict[str, Any]) -> None:
        """Log comprehensive episode summary."""
        self.log_diagnostic(
            category="episode",
            event_type="episode_complete",
            data=episode_data,
            console_summary=f"Episode {episode_data.get('episode', '?')} complete: "
                           f"R={episode_data.get('total_reward', 0.0):.2f}, "
                           f"L={episode_data.get('episode_length', 0)}"
        )
    
    def log_training_milestone(self, milestone_data: Dict[str, Any], milestone_name: str) -> None:
        """Log training milestone (every N episodes, etc.)."""
        self.log_diagnostic(
            category="training",
            event_type="milestone",
            data=milestone_data,
            console_summary=f"Training Milestone: {milestone_name}"
        )
    
    def save_final_summary(self, final_stats: Dict[str, Any]) -> str:
        """Save final training summary and return path."""
        
        # Compile complete summary
        complete_summary = {
            'training_metadata': self.training_metadata,
            'completion_time': datetime.now().isoformat(),
            'final_statistics': final_stats,
            'file_locations': {
                'diagnostics_log': str(self.diagnostics_log_path),
                'json_log': str(self.json_log_path),
                'run_directory': str(self.run_dir)
            }
        }
        
        # Save summary
        with open(self.summary_path, 'w') as f:
            json.dump(complete_summary, f, indent=2, default=str)
        
        self.logger.info(f"Final summary saved to: {self.summary_path}")
        return str(self.summary_path)
    
    def get_output_directory(self) -> str:
        """Get the organized output directory for this training run."""
        return str(self.run_dir)
    
    def get_diagnostics_files(self) -> Dict[str, str]:
        """Get paths to all diagnostics files."""
        return {
            'text_log': str(self.diagnostics_log_path),
            'json_log': str(self.json_log_path), 
            'summary': str(self.summary_path),
            'run_directory': str(self.run_dir)
        }
    
    def flush(self) -> None:
        """Flush all file buffers."""
        if self.log_file:
            self.log_file.flush()
        if self.json_file:
            self.json_file.flush()
        self.buffer_count = 0
    
    def close(self) -> None:
        """Close all file handles and finalize logging."""
        self.flush()
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            
        if self.json_file:
            self.json_file.close()
            self.json_file = None
        
        self.logger.info(f"Diagnostics logging completed. Files in: {self.run_dir}")
    
    def _setup_console_logger(self) -> logging.Logger:
        """Setup console logger for SB3 integration."""
        logger = logging.getLogger(f"m10_diagnostics_{id(self)}")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Remove existing handlers to avoid duplication
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
            formatter = logging.Formatter('%(asctime)s - [M10] - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Prevent propagation to avoid duplication with other loggers
        logger.propagate = False
        
        return logger
    
    def _initialize_logging(self) -> None:
        """Initialize file logging."""
        
        # Initialize text log file
        if self.config.file_output:
            self.log_file = open(self.diagnostics_log_path, 'w')
            self.log_file.write(f"M10 Diagnostics Log - Training Session Started\n")
            self.log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
            self.log_file.write(f"Session ID: {self.config.session_id}\n")
            self.log_file.write(f"Algorithm: {self.config.algorithm}\n")
            self.log_file.write(f"Output Directory: {self.run_dir}\n")
            self.log_file.write("=" * 80 + "\n\n")
        
        # Initialize JSON log file
        if self.config.json_output:
            self.json_file = open(self.json_log_path, 'w')
            
            # Write initial metadata entry
            metadata_entry = {
                'timestamp': datetime.now().isoformat(),
                'category': 'system',
                'event_type': 'logging_initialized',
                'level': 'INFO',
                'data': self.training_metadata
            }
            self.json_file.write(json.dumps(metadata_entry) + "\n")
        
        # Console notification
        self.logger.info(f"M10 diagnostics initialized - Output dir: {self.run_dir}")
    
    def _write_to_log_file(
        self,
        timestamp: datetime,
        category: str,
        event_type: str,
        data: Dict[str, Any],
        level: str
    ) -> None:
        """Write structured entry to text log file."""
        if not self.log_file:
            return
        
        self.log_file.write(f"[{timestamp.strftime('%H:%M:%S')}] {level} - {category.upper()}.{event_type}\n")
        
        # Write data in readable format
        for key, value in data.items():
            if isinstance(value, dict):
                self.log_file.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    self.log_file.write(f"    {sub_key}: {sub_value}\n")
            else:
                self.log_file.write(f"  {key}: {value}\n")
        
        self.log_file.write("\n")
    
    def _write_to_json_file(self, log_entry: Dict[str, Any]) -> None:
        """Write structured entry to JSON lines file."""
        if not self.json_file:
            return
        
        self.json_file.write(json.dumps(log_entry, default=str) + "\n")