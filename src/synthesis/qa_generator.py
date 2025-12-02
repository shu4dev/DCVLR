"""
Q/A Generation module for the Team-1 Data Synthesis Pipeline.
Handles question/answer/reasoning generation using LLMs.
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class QAGenerator:
    """
    Generates Question/Answer/Reasoning triplets using LLMs.
    """

    def __init__(
        self,
        model_name: str = "tiiuae/falcon-7b-instruct",
        config: Optional[Dict] = None,
        device: str = "cuda"
    ):
        """
        Initialize the Q/A generator.

        Args:
            model_name: Name or path of the LLM model
            config: Configuration dictionary
            device: Device to run the model on
        """
        self.model_name = model_name
        self.config = config or {}
        self.device = device
        self.model = None
        self.tokenizer = None

        # Load model lazily to avoid import errors
        self._load_model()

    def _load_model(self):
        """Load the LLM model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set pad_token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None

    def generate(
        self,
        image_features: Dict[str, Any],
        bin_type: str
    ) -> Optional[Dict[str, str]]:
        """
        Generate Q/A/Reasoning for an image based on its features.

        Args:
            image_features: Extracted features from the image
            bin_type: Type of bin (A, B, or C)

        Returns:
            Dictionary with question, answer, and reasoning
        """
        if not self.model or not self.tokenizer:
            # Return placeholder if model not available
            return self._generate_placeholder(image_features, bin_type)

        try:
            # Create prompt based on bin type
            prompt = self._create_prompt(image_features, bin_type)

            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_tokens', 512),
                temperature=self.config.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse the response
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error generating Q/A: {e}")
            return self._generate_placeholder(image_features, bin_type)

    def _create_prompt(self, features: Dict[str, Any], bin_type: str) -> str:
        """Create a prompt for the LLM based on image features and bin type."""
        if bin_type == 'A':
            # Text/Arithmetic focused
            prompt = f"""Based on the following image information, generate a question that requires reading text or performing arithmetic, along with an answer and step-by-step reasoning.

Image features:
- OCR Text: {features.get('ocr_text', 'N/A')}
- Caption: {features.get('caption', 'N/A')}
- Objects: {features.get('objects', [])}

Generate a JSON response with:
- question: A question about the text or requiring calculation
- answer: The correct answer
- reasoning: Step-by-step reasoning to arrive at the answer

Response:"""
        elif bin_type == 'B':
            # Object/Spatial focused
            prompt = f"""Based on the following image information, generate a question about objects and their spatial relationships, along with an answer and reasoning.

Image features:
- Objects detected: {features.get('objects', [])}
- Spatial relations: {features.get('spatial_relations', [])}
- Caption: {features.get('caption', 'N/A')}

Generate a JSON response with:
- question: A question about objects or spatial relationships
- answer: The correct answer
- reasoning: Step-by-step reasoning to arrive at the answer

Response:"""
        else:  # Bin C
            # Commonsense/Attribute focused
            prompt = f"""Based on the following image information, generate a question requiring commonsense reasoning or attribute identification, along with an answer and reasoning.

Image features:
- Caption: {features.get('caption', 'N/A')}
- Scene: {features.get('scene', 'N/A')}
- Attributes: {features.get('attributes', [])}

Generate a JSON response with:
- question: A question requiring commonsense reasoning
- answer: The correct answer
- reasoning: Step-by-step reasoning to arrive at the answer

Response:"""

        return prompt

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract Q/A/Reasoning."""
        import json

        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback parsing
        return {
            'question': 'What is shown in this image?',
            'answer': 'Unable to parse response',
            'reasoning': response
        }

    def _generate_placeholder(
        self,
        features: Dict[str, Any],
        bin_type: str
    ) -> Dict[str, str]:
        """Generate placeholder Q/A when model is not available."""
        if bin_type == 'A':
            question = "What text is visible in the image?"
            answer = features.get('ocr_text', 'No text detected')
            reasoning = f"Looking at the image, I can see the following text: {answer}"
        elif bin_type == 'B':
            objects = features.get('objects', ['unknown objects'])
            question = f"What objects are present in the image?"
            answer = ', '.join(objects) if objects else 'No objects detected'
            reasoning = f"By examining the image, I identified the following objects: {answer}"
        else:
            question = "What is happening in this image?"
            answer = features.get('caption', 'Unable to determine')
            reasoning = f"Based on the image content: {answer}"

        return {
            'question': question,
            'answer': answer,
            'reasoning': reasoning
        }
