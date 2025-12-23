"""
FillProcessor - Created for TRADER 2.0

Handles async fill queue processing for real-time order fills.
Processes WebSocket fill messages asynchronously without blocking the main trading loop.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Deque
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("kalshiflow_rl.trading.services.fill_processor")


@dataclass
class FillEvent:
    """Represents a fill event from WebSocket."""
    kalshi_order_id: str
    timestamp: float
    fill_data: Dict[str, Any]
    processed: bool = False
    processing_attempts: int = 0


class FillProcessor:
    """
    Async fill queue processor.
    
    Receives fill events from WebSocket and processes them asynchronously
    to update orders and positions without blocking other operations.
    """
    
    def __init__(
        self,
        order_service: 'OrderService',
        status_logger: Optional['StatusLogger'] = None,
        websocket_manager=None
    ):
        """
        Initialize FillProcessor.
        
        Args:
            order_service: OrderService instance for order updates
            status_logger: Optional StatusLogger for activity tracking
            websocket_manager: Global WebSocketManager for broadcasting (optional)
        """
        self.order_service = order_service
        self.status_logger = status_logger
        self.websocket_manager = websocket_manager
        
        # Fill queue and processing
        self.fill_queue: Deque[FillEvent] = deque()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_processing = False
        self.stop_processing = False
        
        # Processing statistics
        self.stats = {
            "fills_received": 0,
            "fills_processed": 0,
            "processing_errors": 0,
            "queue_max_size": 0,
            "avg_processing_time": 0.0,
            "last_fill_time": None
        }
        
        # Processing configuration
        self.max_queue_size = 1000
        self.batch_size = 10
        self.processing_interval = 0.1  # Process every 100ms
        self.max_retry_attempts = 3
        
        logger.info("FillProcessor initialized")
    
    async def add_fill_event(self, kalshi_order_id: str, fill_data: Dict[str, Any]) -> bool:
        """
        Add a fill event to the processing queue.
        
        Args:
            kalshi_order_id: Kalshi's order ID
            fill_data: Fill data from WebSocket
            
        Returns:
            True if added successfully, False if queue is full
        """
        try:
            # Check queue size
            if len(self.fill_queue) >= self.max_queue_size:
                logger.warning(f"Fill queue at max capacity ({self.max_queue_size}), dropping fill")
                return False
            
            # Create fill event
            fill_event = FillEvent(
                kalshi_order_id=kalshi_order_id,
                timestamp=time.time(),
                fill_data=fill_data
            )
            
            # Add to queue
            self.fill_queue.append(fill_event)
            
            # Update statistics
            self.stats["fills_received"] += 1
            self.stats["last_fill_time"] = fill_event.timestamp
            self.stats["queue_max_size"] = max(self.stats["queue_max_size"], len(self.fill_queue))
            
            logger.debug(f"Fill event queued: {kalshi_order_id} (queue size: {len(self.fill_queue)})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding fill event: {e}")
            return False
    
    async def _process_fill_event(self, fill_event: FillEvent) -> bool:
        """
        Process a single fill event.
        
        Args:
            fill_event: Fill event to process
            
        Returns:
            True if processed successfully
        """
        try:
            start_time = time.time()
            
            # Process fill through OrderService
            processed_order = self.order_service.handle_fill_event(
                fill_event.kalshi_order_id,
                fill_event.fill_data
            )
            
            processing_time = time.time() - start_time
            
            if processed_order:
                # Mark as processed
                fill_event.processed = True
                
                # Update statistics
                self.stats["fills_processed"] += 1
                
                # Update average processing time
                if self.stats["fills_processed"] > 0:
                    current_avg = self.stats["avg_processing_time"]
                    new_avg = (current_avg * (self.stats["fills_processed"] - 1) + processing_time) / self.stats["fills_processed"]
                    self.stats["avg_processing_time"] = new_avg
                
                # Log activity
                if self.status_logger:
                    await self.status_logger.log_action_result(
                        "fill_processed_async",
                        f"{processed_order.ticker} {processed_order.side.name} {processed_order.quantity}",
                        processing_time
                    )
                
                logger.debug(f"Fill processed: {fill_event.kalshi_order_id} ({processing_time:.3f}s)")
                return True
            
            else:
                # Processing failed
                fill_event.processing_attempts += 1
                logger.warning(f"Fill processing failed: {fill_event.kalshi_order_id} (attempt {fill_event.processing_attempts})")
                
                if fill_event.processing_attempts >= self.max_retry_attempts:
                    logger.error(f"Fill processing failed after {self.max_retry_attempts} attempts: {fill_event.kalshi_order_id}")
                    self.stats["processing_errors"] += 1
                    return True  # Remove from queue
                
                return False  # Keep in queue for retry
            
        except Exception as e:
            fill_event.processing_attempts += 1
            logger.error(f"Error processing fill event {fill_event.kalshi_order_id}: {e}")
            
            if fill_event.processing_attempts >= self.max_retry_attempts:
                logger.error(f"Dropping fill event after {self.max_retry_attempts} attempts: {fill_event.kalshi_order_id}")
                self.stats["processing_errors"] += 1
                return True  # Remove from queue
            
            return False  # Keep in queue for retry
    
    async def _processing_loop(self) -> None:
        """
        Main processing loop that handles the fill queue.
        """
        logger.info("Fill processing loop started")
        
        while not self.stop_processing:
            try:
                # Process fills in batches
                processed_count = 0
                batch_start_time = time.time()
                
                # Process up to batch_size fills
                while processed_count < self.batch_size and self.fill_queue:
                    fill_event = self.fill_queue.popleft()
                    
                    # Process the fill
                    if await self._process_fill_event(fill_event):
                        processed_count += 1
                    else:
                        # Put back at front for retry
                        self.fill_queue.appendleft(fill_event)
                        break  # Stop processing batch if we hit a retry
                
                # Log batch processing if we processed anything
                if processed_count > 0:
                    batch_duration = time.time() - batch_start_time
                    logger.debug(f"Processed {processed_count} fills in {batch_duration:.3f}s")
                
                # Log service status periodically
                if self.status_logger and time.time() % 10 < self.processing_interval:
                    await self.status_logger.log_service_status(
                        "FillProcessor", "processing",
                        {
                            "queue_size": len(self.fill_queue),
                            "fills_processed": self.stats["fills_processed"],
                            "processing_rate": processed_count / max(0.001, batch_duration) if processed_count > 0 else 0
                        }
                    )
                
                # Wait before next processing cycle
                await asyncio.sleep(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in fill processing loop: {e}")
                await asyncio.sleep(1.0)  # Longer wait on error
        
        logger.info("Fill processing loop stopped")
    
    async def start(self) -> bool:
        """
        Start the fill processor.
        
        Returns:
            True if started successfully
        """
        try:
            if self.is_processing:
                logger.warning("Fill processor already running")
                return True
            
            logger.info("Starting fill processor")
            
            # Reset state
            self.stop_processing = False
            self.is_processing = True
            
            # Start processing task
            self.processing_task = asyncio.create_task(self._processing_loop())
            
            # Log service status
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "FillProcessor", "started",
                    {"queue_size": len(self.fill_queue)}
                )
            
            logger.info("Fill processor started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting fill processor: {e}")
            self.is_processing = False
            return False
    
    async def stop(self) -> bool:
        """
        Stop the fill processor.
        
        Returns:
            True if stopped successfully
        """
        try:
            if not self.is_processing:
                logger.info("Fill processor not running")
                return True
            
            logger.info("Stopping fill processor")
            
            # Signal stop
            self.stop_processing = True
            
            # Wait for processing task to complete
            if self.processing_task and not self.processing_task.done():
                try:
                    await asyncio.wait_for(self.processing_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Processing task timeout during stop - cancelling")
                    self.processing_task.cancel()
                    try:
                        await self.processing_task
                    except asyncio.CancelledError:
                        pass
            
            self.is_processing = False
            self.processing_task = None
            
            # Process remaining fills in queue
            remaining_fills = len(self.fill_queue)
            if remaining_fills > 0:
                logger.info(f"Processing {remaining_fills} remaining fills before stop")
                
                # Process remaining fills synchronously
                processed = 0
                while self.fill_queue and processed < 100:  # Limit to prevent hanging
                    fill_event = self.fill_queue.popleft()
                    if await self._process_fill_event(fill_event):
                        processed += 1
                
                logger.info(f"Processed {processed} remaining fills")
            
            # Log service status
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "FillProcessor", "stopped",
                    {"processed_remaining": processed if remaining_fills > 0 else 0}
                )
            
            logger.info("Fill processor stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping fill processor: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fill processor statistics."""
        current_time = time.time()
        
        return {
            "is_processing": self.is_processing,
            "queue_size": len(self.fill_queue),
            "fills_received": self.stats["fills_received"],
            "fills_processed": self.stats["fills_processed"],
            "processing_errors": self.stats["processing_errors"],
            "queue_max_size": self.stats["queue_max_size"],
            "avg_processing_time": self.stats["avg_processing_time"],
            "last_fill_time": self.stats["last_fill_time"],
            "time_since_last_fill": current_time - self.stats["last_fill_time"] if self.stats["last_fill_time"] else None,
            "processing_rate": self.stats["fills_processed"] / max(1, current_time - (self.stats["last_fill_time"] or current_time)) if self.stats["last_fill_time"] else 0.0,
            "error_rate": self.stats["processing_errors"] / max(1, self.stats["fills_received"]),
            "configuration": {
                "max_queue_size": self.max_queue_size,
                "batch_size": self.batch_size,
                "processing_interval": self.processing_interval,
                "max_retry_attempts": self.max_retry_attempts
            }
        }
    
    def clear_queue(self) -> int:
        """
        Clear the fill queue.
        
        Returns:
            Number of fills cleared
        """
        cleared_count = len(self.fill_queue)
        self.fill_queue.clear()
        
        logger.warning(f"Fill queue cleared: {cleared_count} fills removed")
        
        return cleared_count
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status for monitoring."""
        if not self.fill_queue:
            return {"queue_empty": True}
        
        # Analyze queue
        current_time = time.time()
        oldest_fill = min(self.fill_queue, key=lambda f: f.timestamp) if self.fill_queue else None
        newest_fill = max(self.fill_queue, key=lambda f: f.timestamp) if self.fill_queue else None
        
        unprocessed_count = sum(1 for f in self.fill_queue if not f.processed)
        retry_count = sum(1 for f in self.fill_queue if f.processing_attempts > 0)
        
        return {
            "queue_empty": False,
            "queue_size": len(self.fill_queue),
            "unprocessed_fills": unprocessed_count,
            "retrying_fills": retry_count,
            "oldest_fill_age": current_time - oldest_fill.timestamp if oldest_fill else 0,
            "newest_fill_age": current_time - newest_fill.timestamp if newest_fill else 0,
            "queue_span": newest_fill.timestamp - oldest_fill.timestamp if oldest_fill and newest_fill else 0
        }