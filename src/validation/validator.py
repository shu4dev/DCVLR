"""
Data validation module for the Team-1 Data Synthesis Pipeline.
Handles validation and quality control of generated Q/A pairs.
"""

import logging
from typing import Dict, List, Any, Optional
import hashlib

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates and cleans the generated Q/A dataset.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_question_length = self.config.get('min_question_length', 5)
        self.min_answer_length = self.config.get('min_answer_length', 1)
        self.min_reasoning_length = self.config.get('min_reasoning_length', 10)
        self.max_question_length = self.config.get('max_question_length', 200)
        self.max_answer_length = self.config.get('max_answer_length', 500)

    def validate(self, qa_entry: Dict[str, Any]) -> bool:
        """
        Validate a single Q/A entry.

        Args:
            qa_entry: Dictionary containing question, answer, reasoning

        Returns:
            True if entry is valid
        """
        # Check required fields
        if not self._check_required_fields(qa_entry):
            return False

        # Check field lengths
        if not self._check_lengths(qa_entry):
            return False

        # Check content quality
        if not self._check_content_quality(qa_entry):
            return False

        # Check grounding (answer should be derivable from reasoning)
        if not self._check_grounding(qa_entry):
            return False

        return True

    def _check_required_fields(self, qa_entry: Dict[str, Any]) -> bool:
        """Check that all required fields are present."""
        required = ['question', 'answer', 'reasoning']

        for field in required:
            if field not in qa_entry:
                logger.debug(f"Missing required field: {field}")
                return False
            if not qa_entry[field]:
                logger.debug(f"Empty required field: {field}")
                return False

        return True

    def _check_lengths(self, qa_entry: Dict[str, Any]) -> bool:
        """Check that field lengths are within acceptable ranges."""
        question = qa_entry.get('question', '')
        answer = qa_entry.get('answer', '')
        reasoning = qa_entry.get('reasoning', '')

        # Check minimum lengths
        if len(question.split()) < self.min_question_length:
            logger.debug(f"Question too short: {len(question.split())} words")
            return False

        if len(answer.split()) < self.min_answer_length:
            logger.debug(f"Answer too short: {len(answer.split())} words")
            return False

        if len(reasoning.split()) < self.min_reasoning_length:
            logger.debug(f"Reasoning too short: {len(reasoning.split())} words")
            return False

        # Check maximum lengths
        if len(question.split()) > self.max_question_length:
            logger.debug(f"Question too long: {len(question.split())} words")
            return False

        if len(answer.split()) > self.max_answer_length:
            logger.debug(f"Answer too long: {len(answer.split())} words")
            return False

        return True

    def _check_content_quality(self, qa_entry: Dict[str, Any]) -> bool:
        """Check content quality of the Q/A entry."""
        question = qa_entry.get('question', '').lower()
        answer = qa_entry.get('answer', '').lower()

        # Check for placeholder text
        placeholders = ['n/a', 'unknown', 'unable to', 'cannot determine', 'error']
        for placeholder in placeholders:
            if placeholder in answer:
                logger.debug(f"Placeholder text in answer: {placeholder}")
                return False

        # Check that question is actually a question
        if not any([
            question.endswith('?'),
            question.startswith('what'),
            question.startswith('how'),
            question.startswith('why'),
            question.startswith('where'),
            question.startswith('when'),
            question.startswith('who'),
            question.startswith('which'),
            question.startswith('is'),
            question.startswith('are'),
            question.startswith('can'),
            question.startswith('does'),
            question.startswith('do')
        ]):
            logger.debug("Text doesn't appear to be a question")
            return False

        return True

    def _check_grounding(self, qa_entry: Dict[str, Any]) -> bool:
        """
        Check that the answer is grounded in the reasoning.

        The reasoning should logically lead to the answer.
        """
        answer = qa_entry.get('answer', '').lower()
        reasoning = qa_entry.get('reasoning', '').lower()

        # Simple check: answer words should appear in reasoning
        answer_words = set(answer.split())
        reasoning_words = set(reasoning.split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does',
                      'did', 'will', 'would', 'could', 'should', 'may',
                      'might', 'must', 'shall', 'can', 'need', 'dare',
                      'ought', 'used', 'to', 'of', 'in', 'for', 'on',
                      'with', 'at', 'by', 'from', 'as', 'into', 'through',
                      'during', 'before', 'after', 'above', 'below', 'up',
                      'down', 'out', 'off', 'over', 'under', 'again',
                      'further', 'then', 'once', 'and', 'or', 'but', 'if',
                      'because', 'until', 'while', 'that', 'this', 'these',
                      'those', 'it', 'its'}

        answer_words = answer_words - stop_words
        reasoning_words = reasoning_words - stop_words

        if not answer_words:
            return True  # Can't validate if no content words

        # At least some answer words should appear in reasoning
        overlap = answer_words.intersection(reasoning_words)
        overlap_ratio = len(overlap) / len(answer_words)

        if overlap_ratio < 0.3:
            logger.debug(f"Poor grounding: only {overlap_ratio:.2%} overlap")
            return False

        return True

    def remove_duplicates(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate Q/A entries from the dataset.

        Args:
            dataset: List of Q/A entries

        Returns:
            Deduplicated list
        """
        seen_hashes = set()
        unique_entries = []

        for entry in dataset:
            # Create hash from question and answer
            content = f"{entry.get('question', '')}{entry.get('answer', '')}"
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_entries.append(entry)

        removed = len(dataset) - len(unique_entries)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate entries")

        return unique_entries

    def validate_batch(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a batch of Q/A entries.

        Args:
            dataset: List of Q/A entries

        Returns:
            List of valid entries
        """
        valid_entries = []

        for entry in dataset:
            if self.validate(entry):
                valid_entries.append(entry)

        logger.info(f"Validated {len(valid_entries)}/{len(dataset)} entries")
        return valid_entries
