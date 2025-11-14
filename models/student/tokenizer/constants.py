"""
Token ID constants for special tokens.

These IDs correspond to the actual tokenizer encoding (dynamic from tokenizer):
- 0: <unk>
- 1: <s>
- 2: </s>
- 32000: <bot> (beginning of latent span) - actual tokenizer encoding
- 32001: <eot> (end of latent span) - actual tokenizer encoding

Note: The tokenizer_config.json shows IDs 3 and 4 in added_tokens_decoder,
but the actual tokenizer encodes these tokens as 32000 and 32001.
These constants reflect the actual runtime token IDs.
"""
# @author: @darianrosebrook

# Special token IDs (from actual tokenizer encoding)
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
BOT_TOKEN_ID = 32000  # Beginning of latent span (actual tokenizer encoding)
EOT_TOKEN_ID = 32001  # End of latent span (actual tokenizer encoding)

# Token strings
BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"
