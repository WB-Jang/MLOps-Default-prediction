"""Tests for Kafka integration."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.kafka.producer import KafkaDataProducer
from src.kafka.consumer import KafkaDataConsumer, RawDataConsumer


class TestKafkaProducer:
    """Test cases for Kafka producer."""
    
    @patch('src.kafka.producer.KafkaProducer')
    def test_producer_connect(self, mock_kafka_producer):
        """Test producer connection."""
        producer = KafkaDataProducer()
        producer.connect()
        
        mock_kafka_producer.assert_called_once()
        assert producer.producer is not None
    
    @patch('src.kafka.producer.KafkaProducer')
    def test_send_raw_data(self, mock_kafka_producer):
        """Test sending raw data."""
        producer = KafkaDataProducer()
        producer.producer = mock_kafka_producer.return_value
        
        test_data = {"key": "value", "number": 123}
        producer.send_raw_data(test_data, key="test")
        
        producer.producer.send.assert_called_once()
    
    @patch('src.kafka.producer.KafkaProducer')
    def test_send_processed_data(self, mock_kafka_producer):
        """Test sending processed data."""
        producer = KafkaDataProducer()
        producer.producer = mock_kafka_producer.return_value
        
        test_data = {"processed": True, "score": 0.85}
        producer.send_processed_data(test_data, key="test")
        
        producer.producer.send.assert_called_once()
    
    @patch('src.kafka.producer.KafkaProducer')
    def test_send_command(self, mock_kafka_producer):
        """Test sending commands."""
        producer = KafkaDataProducer()
        producer.producer = mock_kafka_producer.return_value
        
        command = {"command": "train", "model_id": "123"}
        producer.send_command(command, key="command")
        
        producer.producer.send.assert_called_once()


class TestKafkaConsumer:
    """Test cases for Kafka consumer."""
    
    @patch('src.kafka.consumer.KafkaConsumer')
    def test_consumer_connect(self, mock_kafka_consumer):
        """Test consumer connection."""
        consumer = RawDataConsumer()
        consumer.connect()
        
        mock_kafka_consumer.assert_called_once()
        assert consumer.consumer is not None
    
    @patch('src.kafka.consumer.KafkaConsumer')
    def test_consume_batch(self, mock_kafka_consumer):
        """Test batch consumption."""
        consumer = KafkaDataConsumer(['test_topic'])
        consumer.consumer = mock_kafka_consumer.return_value
        
        # Mock poll response
        mock_record = Mock()
        mock_record.topic = 'test_topic'
        mock_record.partition = 0
        mock_record.offset = 1
        mock_record.key = b'test_key'
        mock_record.value = {'data': 'test'}
        
        consumer.consumer.poll.return_value = {
            Mock(): [mock_record]
        }
        
        messages = consumer.consume_batch(batch_size=10)
        
        assert len(messages) == 1
        assert messages[0]['value'] == {'data': 'test'}
