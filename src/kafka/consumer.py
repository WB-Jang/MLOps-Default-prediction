"""Kafka consumer for receiving data from Kafka topics."""
import json
from typing import Callable, Dict, Any, Optional, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from loguru import logger

from config.settings import settings


class KafkaDataConsumer:
    """Consumer for receiving data from Kafka topics."""
    
    def __init__(self, topics: List[str], group_id: Optional[str] = None):
        """
        Initialize Kafka consumer.
        
        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID
        """
        self.topics = topics
        self.group_id = group_id or settings.kafka_consumer_group
        self.consumer: Optional[KafkaConsumer] = None
        self.bootstrap_servers = settings.kafka_bootstrap_servers.split(',')
    
    def connect(self) -> None:
        """Establish connection to Kafka and subscribe to topics."""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True
            )
            logger.info(
                f"Connected to Kafka consumer: {self.bootstrap_servers}, "
                f"topics={self.topics}, group_id={self.group_id}"
            )
        except Exception as e:
            logger.error(f"Failed to connect Kafka consumer: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Kafka consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer connection closed")
    
    def consume(
        self,
        handler: Callable[[Dict[str, Any]], None],
        max_messages: Optional[int] = None
    ) -> None:
        """
        Consume messages from subscribed topics.
        
        Args:
            handler: Callback function to process each message
            max_messages: Maximum number of messages to consume (None for infinite)
        """
        if not self.consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        
        try:
            message_count = 0
            logger.info("Starting message consumption...")
            
            for message in self.consumer:
                try:
                    # Process message with handler
                    handler(message.value)
                    
                    message_count += 1
                    if max_messages and message_count >= max_messages:
                        logger.info(f"Reached max_messages limit: {max_messages}")
                        break
                        
                except Exception as e:
                    logger.error(
                        f"Error processing message from {message.topic}: {e}",
                        exc_info=True
                    )
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except KafkaError as e:
            logger.error(f"Kafka error during consumption: {e}")
            raise
    
    def consume_batch(
        self,
        batch_size: int = 100,
        timeout_ms: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Consume a batch of messages.
        
        Args:
            batch_size: Number of messages to consume
            timeout_ms: Timeout in milliseconds
            
        Returns:
            List of consumed messages
        """
        if not self.consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")
        
        messages = []
        try:
            message_batch = self.consumer.poll(timeout_ms=timeout_ms, max_records=batch_size)
            
            for topic_partition, records in message_batch.items():
                for record in records:
                    messages.append({
                        'topic': record.topic,
                        'partition': record.partition,
                        'offset': record.offset,
                        'key': record.key.decode('utf-8') if record.key and isinstance(record.key, bytes) else record.key,
                        'value': record.value
                    })
            
            logger.debug(f"Consumed batch of {len(messages)} messages")
            return messages
            
        except KafkaError as e:
            logger.error(f"Error consuming batch: {e}")
            raise


class RawDataConsumer(KafkaDataConsumer):
    """Consumer specifically for raw_data topic."""
    
    def __init__(self, group_id: Optional[str] = None):
        super().__init__([settings.kafka_raw_data_topic], group_id)


class ProcessedDataConsumer(KafkaDataConsumer):
    """Consumer specifically for processed_data topic."""
    
    def __init__(self, group_id: Optional[str] = None):
        super().__init__([settings.kafka_processed_data_topic], group_id)


class CommandConsumer(KafkaDataConsumer):
    """Consumer specifically for commands topic."""
    
    def __init__(self, group_id: Optional[str] = None):
        super().__init__([settings.kafka_commands_topic], group_id)


# Convenience function to create consumers
def create_consumer(topics: List[str], group_id: Optional[str] = None) -> KafkaDataConsumer:
    """
    Create and return a Kafka consumer.
    
    Args:
        topics: List of topics to subscribe to
        group_id: Consumer group ID
        
    Returns:
        KafkaDataConsumer instance
    """
    return KafkaDataConsumer(topics, group_id)
