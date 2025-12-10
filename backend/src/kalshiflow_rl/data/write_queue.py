"""
Async write queue infrastructure for non-blocking orderbook persistence.

Provides OrderbookWriteQueue with configurable batching, sampling, and
flush intervals to ensure WebSocket message processing never blocks
on database writes.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

from .database import rl_db
from ..config import config

logger = logging.getLogger("kalshiflow_rl.write_queue")


class MessageType(Enum):
    """Types of orderbook messages that can be queued."""
    SNAPSHOT = "snapshot"
    DELTA = "delta"


class OrderbookWriteQueue:
    """
    Async write queue for non-blocking orderbook data persistence.
    
    Features:
    - Non-blocking enqueue operations (always returns immediately)
    - Configurable batching for efficient database writes
    - Sampling for delta messages to reduce volume
    - Automatic flush intervals to ensure data freshness
    - Backpressure handling with queue size limits
    - Graceful shutdown with message preservation
    """
    
    def __init__(
        self,
        batch_size: int = None,
        flush_interval: float = None,
        max_queue_size: int = None
    ):
        """
        Initialize write queue with configuration.
        
        Args:
            batch_size: Number of messages to batch before writing (default from config)
            flush_interval: Seconds between forced flushes (default from config)
            max_queue_size: Maximum queue size before backpressure (default from config)
        """
        self.batch_size = batch_size if batch_size is not None else config.ORDERBOOK_QUEUE_BATCH_SIZE
        self.flush_interval = flush_interval if flush_interval is not None else config.ORDERBOOK_QUEUE_FLUSH_INTERVAL
        self.max_queue_size = max_queue_size if max_queue_size is not None else config.ORDERBOOK_MAX_QUEUE_SIZE
        
        # Async queues for different message types
        self._snapshot_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size // 4)
        self._delta_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Background task management
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Session tracking
        self._session_id: Optional[int] = None
        
        # Metrics and monitoring
        self._messages_enqueued = 0
        self._messages_written = 0
        self._snapshots_written = 0
        self._deltas_written = 0
        self._last_flush_time = time.time()
        self._queue_full_errors = 0
        
        
        logger.info(
            f"OrderbookWriteQueue initialized: batch_size={self.batch_size}, "
            f"flush_interval={self.flush_interval}s, max_queue_size={self.max_queue_size}"
        )
    
    def set_session_id(self, session_id: int) -> None:
        """Set the current session ID for all writes."""
        self._session_id = session_id
        logger.info(f"OrderbookWriteQueue session set to {session_id}")
    
    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            logger.warning("OrderbookWriteQueue is already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("OrderbookWriteQueue started")
    
    async def stop(self) -> None:
        """Stop the flush loop and process remaining messages."""
        if not self._running:
            return
        
        logger.info("Stopping OrderbookWriteQueue...")
        self._running = False
        self._shutdown_event.set()
        
        if self._flush_task:
            await self._flush_task
        
        # Flush remaining messages
        await self._flush_all_queues()
        
        logger.info(
            f"OrderbookWriteQueue stopped. Final stats: "
            f"enqueued={self._messages_enqueued}, written={self._messages_written}, "
            f"snapshots={self._snapshots_written}, deltas={self._deltas_written}"
        )
    
    async def enqueue_snapshot(self, snapshot_data: Dict[str, Any]) -> bool:
        """
        Enqueue an orderbook snapshot for writing.
        
        Args:
            snapshot_data: Snapshot data dict with all required fields
            
        Returns:
            bool: True if enqueued successfully, False if queue full
        """
        try:
            # Add message type and timestamp
            message = {
                "type": MessageType.SNAPSHOT.value,
                "data": snapshot_data,
                "enqueued_at": time.time()
            }
            
            # Non-blocking put - will raise QueueFull if full
            self._snapshot_queue.put_nowait(message)
            self._messages_enqueued += 1
            
            return True
            
        except asyncio.QueueFull:
            self._queue_full_errors += 1
            logger.warning(
                f"Snapshot queue full (size: {self._snapshot_queue.qsize()}), "
                f"dropping message for {snapshot_data.get('market_ticker', 'unknown')}"
            )
            return False
        except Exception as e:
            logger.error(f"Error enqueuing snapshot: {e}")
            return False
    
    async def enqueue_delta(self, delta_data: Dict[str, Any]) -> bool:
        """
        Enqueue an orderbook delta for writing (with sampling).
        
        Args:
            delta_data: Delta data dict with all required fields
            
        Returns:
            bool: True if enqueued successfully, False if sampled out or queue full
        """
        try:
            # Add message type and timestamp
            message = {
                "type": MessageType.DELTA.value,
                "data": delta_data,
                "enqueued_at": time.time()
            }
            
            # Non-blocking put - will raise QueueFull if full
            self._delta_queue.put_nowait(message)
            self._messages_enqueued += 1
            
            return True
            
        except asyncio.QueueFull:
            self._queue_full_errors += 1
            logger.warning(
                f"Delta queue full (size: {self._delta_queue.qsize()}), "
                f"dropping message for {delta_data.get('market_ticker', 'unknown')}"
            )
            return False
        except Exception as e:
            logger.error(f"Error enqueuing delta: {e}")
            return False
    
    async def _flush_loop(self) -> None:
        """Main flush loop that processes queued messages."""
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
                await self._flush_all_queues()
            
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry
        
        logger.info("Flush loop stopped")
    
    async def _flush_all_queues(self) -> None:
        """Flush all queues in batch operations."""
        # Process both queues concurrently for efficiency
        await asyncio.gather(
            self._flush_snapshot_queue(),
            self._flush_delta_queue(),
            return_exceptions=True
        )
        
        self._last_flush_time = time.time()
    
    async def _flush_snapshot_queue(self) -> None:
        """Flush snapshot queue in batches."""
        snapshots = []
        
        # Collect messages up to batch size
        while len(snapshots) < self.batch_size and not self._snapshot_queue.empty():
            try:
                message = self._snapshot_queue.get_nowait()
                snapshots.append(message["data"])
            except asyncio.QueueEmpty:
                break
        
        if snapshots and self._session_id:
            try:
                count = await rl_db.batch_insert_snapshots(snapshots, self._session_id)
                self._snapshots_written += count
                self._messages_written += count
                
                if count > 0 and self._snapshots_written % 100 == 0:
                    logger.info(f"Snapshot write checkpoint: {self._snapshots_written} total snapshots written")
                
            except Exception as e:
                logger.error(f"Failed to write {len(snapshots)} snapshots: {e}")
                # Re-queue messages for retry (but limit retries to prevent infinite loop)
                for snapshot in snapshots[:10]:  # Only retry first 10 to prevent memory issues
                    try:
                        await self.enqueue_snapshot(snapshot)
                    except Exception:
                        pass  # Drop message if re-queue fails
        elif snapshots and not self._session_id:
            logger.warning(f"Cannot write {len(snapshots)} snapshots: no session ID set")
    
    async def _flush_delta_queue(self) -> None:
        """Flush delta queue in batches."""
        deltas = []
        
        # Collect messages up to batch size
        while len(deltas) < self.batch_size and not self._delta_queue.empty():
            try:
                message = self._delta_queue.get_nowait()
                deltas.append(message["data"])
            except asyncio.QueueEmpty:
                break
        
        if deltas and self._session_id:
            try:
                count = await rl_db.batch_insert_deltas(deltas, self._session_id)
                self._deltas_written += count
                self._messages_written += count
                
                if count > 0 and self._deltas_written % 1000 == 0:
                    logger.info(f"Delta write checkpoint: {self._deltas_written} total deltas written")
                
            except Exception as e:
                logger.error(f"Failed to write {len(deltas)} deltas: {e}")
                # Re-queue messages for retry (but limit retries)
                for delta in deltas[:10]:  # Only retry first 10
                    try:
                        await self.enqueue_delta(delta)
                    except Exception:
                        pass  # Drop message if re-queue fails
        elif deltas and not self._session_id:
            logger.warning(f"Cannot write {len(deltas)} deltas: no session ID set")
    
    # Monitoring and diagnostics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            "running": self._running,
            "messages_enqueued": self._messages_enqueued,
            "messages_written": self._messages_written,
            "snapshots_written": self._snapshots_written,
            "deltas_written": self._deltas_written,
            "queue_full_errors": self._queue_full_errors,
            "snapshot_queue_size": self._snapshot_queue.qsize(),
            "delta_queue_size": self._delta_queue.qsize(),
            "last_flush_time": self._last_flush_time,
            "config": {
                "batch_size": self.batch_size,
                "flush_interval": self.flush_interval,
                "max_queue_size": self.max_queue_size
            },
            "session_id": self._session_id
        }
    
    def is_healthy(self) -> bool:
        """Check if queue is healthy and processing normally."""
        # Check if queues are not completely full
        snapshot_full = self._snapshot_queue.qsize() >= self._snapshot_queue.maxsize * 0.9
        delta_full = self._delta_queue.qsize() >= self._delta_queue.maxsize * 0.9
        
        # Check if recent flush happened
        time_since_flush = time.time() - self._last_flush_time
        flush_overdue = time_since_flush > (self.flush_interval * 3)
        
        return self._running and not snapshot_full and not delta_full and not flush_overdue
    
    async def force_flush(self) -> None:
        """Force an immediate flush of all queues."""
        await self._flush_all_queues()
        logger.info("Forced flush completed")


# Import config at module level for initialization
from ..config import config

# Global write queue instance - initialized immediately like orderbook_client
write_queue = OrderbookWriteQueue(
    batch_size=config.ORDERBOOK_QUEUE_BATCH_SIZE,
    flush_interval=config.ORDERBOOK_QUEUE_FLUSH_INTERVAL,
    max_queue_size=config.ORDERBOOK_MAX_QUEUE_SIZE
)

def get_write_queue() -> OrderbookWriteQueue:
    """Get the global write queue instance (for backwards compatibility)."""
    return write_queue