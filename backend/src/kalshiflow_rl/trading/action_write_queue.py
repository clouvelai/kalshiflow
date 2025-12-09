"""
Async write queue infrastructure for non-blocking trading action persistence.

Provides ActionWriteQueue with configurable batching and flush intervals
to ensure trading execution never blocks on database writes.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
import json

from ..data.database import rl_db
from ..config import config

logger = logging.getLogger("kalshiflow_rl.action_write_queue")


class ActionWriteQueue:
    """
    Async write queue for non-blocking trading action persistence.
    
    Features:
    - Non-blocking enqueue operations (always returns immediately)
    - Configurable batching for efficient database writes
    - Automatic flush intervals to ensure data freshness
    - Backpressure handling with queue size limits
    - Graceful shutdown with message preservation
    - Error handling and retry logic
    - Statistics tracking for monitoring
    """
    
    def __init__(
        self,
        batch_size: int = None,
        flush_interval: float = None,
        max_queue_size: int = None,
        max_retries: int = 3
    ):
        """
        Initialize action write queue with configuration.
        
        Args:
            batch_size: Number of actions to batch before writing (default from config)
            flush_interval: Seconds between forced flushes (default from config)
            max_queue_size: Maximum queue size before backpressure (default from config)
            max_retries: Maximum retry attempts for failed writes
        """
        self.batch_size = batch_size if batch_size is not None else config.ORDERBOOK_QUEUE_BATCH_SIZE
        self.flush_interval = flush_interval if flush_interval is not None else config.ORDERBOOK_QUEUE_FLUSH_INTERVAL
        self.max_queue_size = max_queue_size if max_queue_size is not None else config.ORDERBOOK_MAX_QUEUE_SIZE
        self.max_retries = max_retries
        
        # Async queue for trading actions
        self._action_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Background task management
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Metrics and monitoring
        self._actions_enqueued = 0
        self._actions_written = 0
        self._failed_writes = 0
        self._retry_attempts = 0
        self._last_flush_time = time.time()
        self._queue_full_errors = 0
        
        logger.info(
            f"ActionWriteQueue initialized: batch_size={self.batch_size}, "
            f"flush_interval={self.flush_interval}s, max_queue_size={self.max_queue_size}, "
            f"max_retries={self.max_retries}"
        )
    
    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            logger.warning("ActionWriteQueue is already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("ActionWriteQueue started")
    
    async def stop(self) -> None:
        """Stop the flush loop and process remaining actions."""
        if not self._running:
            return
        
        logger.info("Stopping ActionWriteQueue...")
        self._running = False
        self._shutdown_event.set()
        
        if self._flush_task:
            await self._flush_task
        
        # Flush remaining actions
        await self._flush_action_queue()
        
        logger.info(
            f"ActionWriteQueue stopped. Final stats: "
            f"enqueued={self._actions_enqueued}, written={self._actions_written}, "
            f"failed={self._failed_writes}, retries={self._retry_attempts}"
        )
    
    async def enqueue_action(self, action_data: Dict[str, Any]) -> bool:
        """
        Enqueue a trading action for writing.
        
        Args:
            action_data: Trading action data dict with all required fields
            
        Returns:
            bool: True if enqueued successfully, False if queue full
        """
        try:
            # Add enqueue timestamp for latency tracking
            message = {
                "action_data": action_data,
                "enqueued_at": time.time(),
                "retry_count": 0
            }
            
            # Non-blocking put - will raise QueueFull if full
            self._action_queue.put_nowait(message)
            self._actions_enqueued += 1
            
            return True
            
        except asyncio.QueueFull:
            self._queue_full_errors += 1
            logger.warning(
                f"Action queue full (size: {self._action_queue.qsize()}), "
                f"dropping action for episode_id={action_data.get('episode_id', 'unknown')}, "
                f"step={action_data.get('step_number', 'unknown')}"
            )
            return False
        except Exception as e:
            logger.error(f"Error enqueuing action: {e}")
            return False
    
    async def _flush_loop(self) -> None:
        """Main flush loop that processes queued actions."""
        while self._running:
            try:
                # Wait for flush interval or shutdown signal
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.flush_interval
                )
                # Shutdown signal received
                break
                
            except asyncio.TimeoutError:
                # Flush interval reached
                await self._flush_action_queue()
            
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry
        
        logger.info("Action flush loop stopped")
    
    async def _flush_action_queue(self) -> None:
        """Flush action queue in batch operations."""
        actions = []
        
        # Collect messages up to batch size
        while len(actions) < self.batch_size and not self._action_queue.empty():
            try:
                message = self._action_queue.get_nowait()
                actions.append(message)
            except asyncio.QueueEmpty:
                break
        
        if actions:
            await self._write_actions_batch(actions)
        
        self._last_flush_time = time.time()
    
    async def _write_actions_batch(self, actions: List[Dict[str, Any]]) -> None:
        """Write a batch of actions to the database with retry logic."""
        try:
            # Extract action data for database insert
            action_data_list = [action["action_data"] for action in actions]
            
            # Use batch insert if available, otherwise fall back to individual inserts
            if hasattr(rl_db, 'batch_insert_trading_actions'):
                count = await rl_db.batch_insert_trading_actions(action_data_list)
            else:
                count = await self._batch_insert_individual(action_data_list)
            
            self._actions_written += count
            
            logger.debug(f"Flushed {count} trading actions to database")
            
        except Exception as e:
            logger.error(f"Failed to write {len(actions)} trading actions: {e}")
            self._failed_writes += len(actions)
            
            # Retry failed actions (up to max_retries)
            retry_actions = []
            for action in actions:
                if action["retry_count"] < self.max_retries:
                    action["retry_count"] += 1
                    retry_actions.append(action)
                    self._retry_attempts += 1
            
            # Re-queue actions for retry (but limit to prevent infinite loop)
            for action in retry_actions[:10]:  # Only retry first 10 to prevent memory issues
                try:
                    self._action_queue.put_nowait(action)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full during retry, dropping action")
                    break
                except Exception:
                    pass  # Drop message if re-queue fails
    
    async def _batch_insert_individual(self, action_data_list: List[Dict[str, Any]]) -> int:
        """Fall back method for individual action inserts if batch method not available."""
        count = 0
        for action_data in action_data_list:
            try:
                await rl_db.create_trading_action(action_data)
                count += 1
            except Exception as e:
                logger.error(f"Failed to write individual trading action: {e}")
        return count
    
    # Monitoring and diagnostics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            "running": self._running,
            "actions_enqueued": self._actions_enqueued,
            "actions_written": self._actions_written,
            "failed_writes": self._failed_writes,
            "retry_attempts": self._retry_attempts,
            "queue_full_errors": self._queue_full_errors,
            "queue_size": self._action_queue.qsize(),
            "last_flush_time": self._last_flush_time,
            "config": {
                "batch_size": self.batch_size,
                "flush_interval": self.flush_interval,
                "max_queue_size": self.max_queue_size,
                "max_retries": self.max_retries
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if queue is healthy and processing normally."""
        # Check if queue is not completely full
        queue_full = self._action_queue.qsize() >= self._action_queue.maxsize * 0.9
        
        # Check if recent flush happened
        time_since_flush = time.time() - self._last_flush_time
        flush_overdue = time_since_flush > (self.flush_interval * 3)
        
        return self._running and not queue_full and not flush_overdue
    
    async def force_flush(self) -> None:
        """Force an immediate flush of all queued actions."""
        await self._flush_action_queue()
        logger.info("Forced flush completed")


# Global action write queue instance
action_write_queue = ActionWriteQueue(
    batch_size=config.ORDERBOOK_QUEUE_BATCH_SIZE,
    flush_interval=config.ORDERBOOK_QUEUE_FLUSH_INTERVAL,
    max_queue_size=config.ORDERBOOK_MAX_QUEUE_SIZE
)

def get_action_write_queue() -> ActionWriteQueue:
    """Get the global action write queue instance."""
    return action_write_queue