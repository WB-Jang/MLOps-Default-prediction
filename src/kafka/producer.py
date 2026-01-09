"""Kafka producer for sending data to Kafka topics."""
import json
from typing import Dict, Any, Optional, List
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger

from config.settings import settings


class KafkaDataProducer:
    """Producer for sending data to Kafka topics."""
    
    def __init__(self):
        """Initialize Kafka producer."""
        self.producer: Optional[KafkaProducer] = None
        self.bootstrap_servers = settings.kafka_bootstrap_servers.split(',')
        
    def connect(self) -> None:
        """Establish connection to Kafka."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            logger.info(f"Connected to Kafka: {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Kafka producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer connection closed")
    
    def send_raw_data(self, data: Dict[str, Any], key: Optional[str] = None) -> None:
        """
        Send raw data to the raw_data topic.
        
        Args:
            data: Data to send
            key: Optional message key for partitioning
        """
        self._send_message(settings.kafka_raw_data_topic, data, key)
    
    def send_processed_data(self, data: Dict[str, Any], key: Optional[str] = None) -> None:
        """
        Send processed data to the processed_data topic.
        
        Args:
            data: Processed data to send
            key: Optional message key for partitioning
        """
        self._send_message(settings.kafka_processed_data_topic, data, key)
    
    def send_command(self, command: Dict[str, Any], key: Optional[str] = None) -> None:
        """
        Send command/event to the commands topic.
        
        Args:
            command: Command or event data
            key: Optional message key for partitioning
        """
        self._send_message(settings.kafka_commands_topic, command, key)
    
    def _send_message(
        self,
        topic: str,
        data: Dict[str, Any],
        key: Optional[str] = None
    ) -> None:
        """
        Send a message to a Kafka topic.
        
        Args:
            topic: Kafka topic name
            data: Data to send
            key: Optional message key
        """
        if not self.producer:
            raise RuntimeError("Producer not connected. Call connect() first.")
        
        try:
            key_bytes = key.encode('utf-8') if key else None
            future = self.producer.send(topic, value=data, key=key_bytes)
            
            # Block until message is sent
            record_metadata = future.get(timeout=10)
            logger.debug(
                f"Message sent to topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}"
            )
        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            raise
    
    def send_batch(self, topic: str, messages: List[Dict[str, Any]]) -> None:
        """
        Send a batch of messages to a Kafka topic.
        
        Args:
            topic: Kafka topic name
            messages: List of messages to send
        """
        if not self.producer:
            raise RuntimeError("Producer not connected. Call connect() first.")
        
        try:
            for msg in messages:
                self.producer.send(topic, value=msg)
            
            self.producer.flush()
            logger.info(f"Sent batch of {len(messages)} messages to {topic}")
        except KafkaError as e:
            logger.error(f"Failed to send batch to {topic}: {e}")
            raise


# Global producer instance
kafka_producer = KafkaDataProducer()
