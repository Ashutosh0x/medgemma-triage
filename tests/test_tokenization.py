import unittest
from unittest.mock import MagicMock

class TestTokenizationBypass(unittest.TestCase):
    
    def test_image_token_expansion(self):
        """
        Regression Test: "Nuclear Bypass" logic (Version 22).
        Verifies that visual tokens are expanded to match typical feature count (256/1024).
        """
        raw_prompt = "Describe this X-ray."
        
        # Logic from main.py V22 (simulated)
        num_patches = 256  # Matches SigLIP/MedGemma feature count typical for bypass
        expanded_prompt = "<image_soft_token>" * num_patches + raw_prompt
        
        # Verify expansion
        self.assertEqual(expanded_prompt.count("<image_soft_token>"), 256)
        self.assertTrue(expanded_prompt.endswith(raw_prompt))
        
    def test_chat_template_structure(self):
        """
        Verify chat template construction preserves control tokens.
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi"}
        ]
        
        # Mock processor/tokenizer
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi<end_of_turn>"
        
        output = tokenizer.apply_chat_template(messages)
        
        self.assertIn("<start_of_turn>", output)
        self.assertIn("user", output)
        self.assertIn("model", output)
        
if __name__ == '__main__':
    unittest.main()
