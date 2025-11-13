"""
Token ID constants for special tokens.

These IDs correspond to the tokenizer_config.json added_tokens_decoder mapping:
- 0: <unk>
- 1: <s>
- 2: </s>
- 3: <bot> (beginning of latent span)
- 4: <eot> (end of latent span)
"""
# @author: @darianrosebrook

# Special token IDs (from tokenizer_config.json added_tokens_decoder)
UNK_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
BOT_TOKEN_ID = 3  # Beginning of latent span
EOT_TOKEN_ID = 4  # End of latent span

# Token strings
BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"

